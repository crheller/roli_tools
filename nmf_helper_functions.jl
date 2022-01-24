using NaNMath, ImageSegmentation, ProgressMeter, LinearAlgebra

function nancorrcoef(x, y)
    # pearson r is the mean of the products of the standard scores (z-scores)
    x_z = (x .- NaNMath.mean(x)) / NaNMath.std(x)
    y_z = (y .- NaNMath.mean(y)) / NaNMath.std(y)
    cc = (1 / (sum(isnan.(x.*y).==false)-1)) * NaNMath.sum(x_z .* y_z)
    return round(cc, sigdigits=5)
end

function unpackh5(ds, filename; raw=true)
    fh = h5open(ds, filename; raw=raw)
    saved_variables = keys(fh);
    res = Dict();
    for key in saved_variables
        res[key] = read(fh, key);
    end
    close(fh) 
    return res
end

function get_brain_region_mask(img, seeds)
    """
    img is a preprocessed 'max' projections
    seeds is a list of tuples: (CartesianIndices, segidx) that lay on the region you want to find
        for now, segidx should always be 1. 
        TODO: Could expand this to mask multiple regions in one go in the future
    """
    segments = seeded_region_growing(img, seeds)
    brain_mask = map(i->segment_mean(segments,i), labels_map(segments)').==segment_mean(segments,1); 
    return brain_mask
end

function rm_corr_rois(X, roi_x, roi_y, z_all; cc_thresh=0.7, xy_spatial_threshold=5, z_spatial_threshold=3)
    """
    remove rois that are highly correlated in time and spatially close together 
    """
    spatialThreshold = xy_spatial_threshold
    zThreshold = z_spatial_threshold
    ccThresh = cc_thresh
    nPutCells = size(X)[2]
    dropidx = zeros(0)
    keepidx = zeros(0)
    @showprogress for c1 in 1:nPutCells
        # get all cells close to this one
        x = roi_x[c1]
        y = roi_y[c1]
        z = z_all[c1]
        candidates = findall( (abs.(roi_x.-x).<spatialThreshold) .& (abs.(roi_y.-y).<spatialThreshold) .& (abs.(z_all.-z).<zThreshold) )
        candidates = [c for c in candidates if ((c in dropidx) == false) & ((c in keepidx) == false)]

        # get correlation of each with c1
        if (length(candidates) > 0) & ((c1 in keepidx)==false) & ((c1 in dropidx)==false)
            cc = Array{Float64}(undef, length(candidates))
            for (i, c2) in enumerate(candidates)
                cc[i] = nancorrcoef(X[:, c1], X[:, c2])
            end

            # find potential indexes to add to dropidx (if correlation is very high with c1)
            ridx = candidates[findall((cc.>ccThresh))]
            # select which to keep based on how many NaN values each has
            if length(ridx)>0
                # find the 'best' of these traces to keep
                missing_values = [sum(isnan.(X[:, ri])) for ri in ridx]
                gidx = ridx[findmin(missing_values)[2]]
                append!(keepidx, gidx)
                append!(dropidx, [ri for ri in ridx if (ri!=gidx) & ((ri in dropidx)==false)])
            else
                # TODO if nothing was this correlated with it, it might not be a cell. See if it has and "signal"
                if ((c1 in keepidx) == false) & ((c1 in dropidx) == false)
                    append!(keepidx, c1) 
                end
            end    
        else
            if ((c1 in keepidx) == false) & ((c1 in dropidx) == false)
                append!(keepidx, c1)
            end
        end
    end
    return Int64.(keepidx)
end

function build_finite_matrix(X; tolerance=1, fs=2)
    """
    Goal is to make a full matrix with no Nans so that we can perform dim reduction etc.
        X is neuron x time
        tolerance specifies time window (in sec) that we'll allow padding. Longer bouts of NaNs are exluded
        fs is sampling rate
    
    returns the new matrix, along with the indices of the time points that we kept
    """
    tolerance = tolerance * fs
    regMat = copy(X)
    dropidx = zeros(0)
    padidx = zeros(0)
    goodidx = zeros(0)
    step = 1
    @showprogress for t in Int64.(1:step:size(X)[2])
        tSlice = zeros(tolerance, size(X)[1])
        nanIdx = zeros(tolerance, size(X)[1])
        times = zeros(0)
        lim = tolerance > (size(X)[2]-t) ? step : tolerance
        for _t in 1:lim
            tSlice[_t, :] = X[:, t+(_t-1)]
            nanIdx[_t, :] = isnan.(tSlice[_t, :])  
            append!(times, t+_t-1)
        end
        times = Int64.(times)
        drop = sum(nanIdx.==true, dims=1).==lim
        pad = (sum(nanIdx.==true, dims=1).>0) .& (sum(nanIdx.==true, dims=1).<lim)
        pass = sum(nanIdx.==true, dims=1).==0
        if sum(drop.==true)>0
            # dropidx has "final" say. If something gets in here, it will be dropped no matter how padding happens
            append!(dropidx, times)
        elseif (sum(pad.==true)>0) & ((t in dropidx)==false)
            append!(padidx, times)
            # this means that for some of the neurons, they can be padded (and others are good as is)
            # for each neuron, figure out which way to pad
            for nidx in Int64.(1:size(X)[1])
                tofillidx = Int64.(findall(isnan.(tSlice[:, nidx]))).-1
                for fi in tofillidx
                    thisidx = fi
                    # replace loop w/ vectorized version. Can be a bit slower (if tol very high), but much cleaner code
                    not_nan = findall(isnan.(X[nidx, t:t+lim]).==false)
                    pad_idx = not_nan[findall(==(minimum(abs.(not_nan.-thisidx))), abs.(not_nan.-thisidx))][1]
                    regMat[nidx, t+thisidx] = X[nidx, t+(pad_idx-1)]
                end     
            end
        elseif sum(pass.==true)==size(X)[1]
            append!(goodidx, times)
        end
    end
    keep = [idx for idx in range(1, size(X)[2], step=1) if (idx in unique(dropidx)).==false]
    regMat = regMat[:, keep]
    return keep, regMat    
end

# ================ PCA =====================
mutable struct PCA_structure
    X::Array{Float32}
    n_components::Int64
    center::Bool
    components_::Array{Float32}
    explained_variance_ratio_::Array{Float32}
end
# constructor to allow for default values
function PCA(;X=zeros(0), n_components=0, center=true, components_=zeros(0), explained_variance_ratio_=zeros(0))
   return PCA_structure(X, n_components, center, components_, explained_variance_ratio_) 
end


# functions to perform PCA on the structure
function fit(self::PCA_structure)
    """
    simple PCA implementation using eigendecomposition of the covariance matrix
    X should be neuron x time
    return none -- modifies PCA object
    """
    isempty(self.X)==false ? nothing : error("Must specify fit matrix 'X'");
    self.n_components = self.n_components==0 ? size(self.X)[1] : self.n_components;
    
    # center data
    if self.center
       self.X = self.X .- mean(self.X, dims=2);
    end
    
    # compute covariance matrix
    cc = cov(self.X');
    
    # eigen-decomposition
    decomp = eigen(cc);
    evecs = decomp.vectors;
    evals = decomp.values;
    
    # sort evals / evecs
    sort = sortperm(decomp.values, rev=true);
    evecs = evecs[:, sort];
    evals = evals[sort];
    
    self.explained_variance_ratio_ = evals[1:self.n_components] ./ sum(evals);
    self.components_ = evecs[:, 1:self.n_components];
    return nothing    
end

function fit_transform(self::PCA_structure)
    """
    simple PCA implementation using eigendecomposition of the covariance matrix
    X should be neuron x time
    returns projection of X onto n_components_ number of PCs
    """
    fit(self)
    return (self.X' * self.components_[:, 1:self.n_components])'
end
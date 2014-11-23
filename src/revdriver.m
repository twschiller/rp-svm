function revdriver( inFile, outFile, nFold, nPart,maxFeatures, method)
   %% Deployment conversions
   if isdeployed
      nFold = str2double(nFold);
      nPart = str2double(nPart);
      maxFeatures = str2double(maxFeatures);
   end

   %% Check arguments
   nFold = uint8(nFold); nPart = uint8(nPart);
   assert(nFold >= 2);
   assert(nPart >= 1);
   assert(strcmp(method,'rm') || strcmp(method,'loo'));

   %% Load workspace
   load(inFile);
   assert(exist('X','var') && exist('Y','var'));
   assert(size(X,1) == size(Y,1));
   assert(isvector(Y));

   %% Scale data
   sX = (X - repmat(min(X,[],1),size(X,1),1)) * ...
      spdiags(1./(max(X,[],1)-min(X,[],1))',0,size(X,2),size(X,2));

   %% Build folds
   folds = revgenfolds(sX,Y,nFold);
   models = cell(nFold,1);

   for f=1:nFold
   
      partSize = uint32(sum(folds{f}.train.Y > 0) * 2);
   
      parts = revpartition(folds{f}.train, nPart, partSize);
   
      features = cell(nPart,1);
      for p=1:nPart
         fprintf('Fold %d - Part %d \n',f,p);
         features{p} = revfs(parts{p},folds{f},method, nFold,maxFeatures);
      end
    
      models{f} = struct('parts',{parts},'features',{features});
   
      clear p partSize parts features;
   end

   clear f;
   save(outFile);
end

function folds = revgenfolds(X,Y,nFold)
   L = size(Y,1);

   folds = cell(nFold,1);
   ip = randperm(L);
   pf = ceil(L / nFold);

   for i=1:nFold
      first = (i-1) * pf + 1;
      last = min([(pf * i) L]);
   
      %% Build test set
      ids = ip(first:last);
      test = struct('X', X(ids,:), 'Y', Y(ids), 'ids', ids);
   
      %% Build train set
      ids = ip(setdiff(1:L, first:last));
      train = struct('X', X(ids,:), 'Y', Y(ids), 'ids', ids);
   
      folds{i} = struct('train',train, 'test',test);
   
      %% Verify mutal exclusion / completeness
      assert(isempty(intersect(train.ids,test.ids)));
      assert(isempty(setdiff(1:L,union(train.ids,test.ids))));
   end
end


function [ parts ] = revpartition( data, nParts, partSize)
%Partition into nParts parts of size partSize
%  Create nParts balanced parts of size partSize by randomly sampling
%  positive and negative instances from the data.

    %% Input validation
    error(nargchk(3, 3, nargin));         
    error(nargoutchk(1, 1, nargout));
    
    X = data.X; Y = data.Y; refs = data.ids;
    
    assert(size(Y,2) == 1);
    assert(size(X,1) == size(Y,1));
    assert(mod(partSize,2) == 0, 'Partition size must be even');
    
    %% Gather information about the input
    [L,F] = size(X);
    
    %the original indexes of the positive/negative indexes
    idxP = find(Y(:,1) > 0); nP = size(idxP,1);
    idxN = find(Y(:,1) < 0); nN = size(idxN,1);
    
    %% Generate the partitions
    parts = cell(nParts,1);
    
    for p=1:nParts
      %% Pre-allocate the partition data
      parts{p} = struct('Y', zeros(partSize,1), 'X', zeros(partSize,F), 'ids', zeros(1,partSize));   
      
      perm = randperm(partSize);%to assign the instances in random order
      
      %% Add the positive instances
      for i=1:partSize/2
          ri = floor(nP .* rand) + 1;%random index
          ai = perm(i);
          parts{p}.X(ai,:) = X(idxP(ri),:);
          parts{p}.Y(ai,:) = Y(idxP(ri),:);
          parts{p}.ids(ai) = refs(idxP(ri));
      end
    
      %% Add the negative instances
      for i=1:partSize/2
          ai = perm(i + partSize / 2);
          ri = floor(nN .* rand) + 1;
          parts{p}.X(ai,:) = X(idxN(ri),:);
          parts{p}.Y(ai,:) = Y(idxN(ri),:);
          parts{p}.ids(ai) = refs(idxN(ri));
      end   
    end
end

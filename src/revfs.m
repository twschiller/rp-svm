function bestfs = revfs( part, fold, method, nFold, maxFeatures )

   [L,F] = size(part.X);
   valid = activeset(part.X);
   
   best = struct('features',[], 'auc',-inf);
   
   change = true;
   
   while change
      change = false;
      
      %% Perform substitution step
      %if size(best.features,1) >= 3
         for f=1:F
            if any(best.features == f) || ~valid(f), continue, end;
            if size(best.features,1)==1 && f <= best.features(1), continue, end;
            
            %% Randomly substitute a feature
            features = best.features;
            ri = floor(rand * size(features,1)) + 1;
            features(ri) = f;
            features = sort(features);
         
            %% Calculate AUC, updating best if necessary
            auc = revevalfs(part,fold,features,nFold,method);
         
            if (auc > best.auc)
               best.features = features;
               best.auc = auc;
              
               fprintf('New best AUC: %f\n',best.auc);
               disp(features);
              
               change = true;
               break;
            end
         end
      %end
   
   
      %% Perform addition step
      if ~change && size(best.features,1) <= maxFeatures
         for f=1:F
            if any(best.features == f) || ~valid(f), continue, end;
         
            features = sort([best.features ; f]);
         
            %% Calculate AUC, updating best if necessary
            auc = revevalfs(part,fold,features,nFold,method);
            
            if (auc > best.auc)
               best.features = features;
               best.auc = auc;
               fprintf('New best AUC: %f\n',best.auc);
               disp(features);
               change = true;
               break;
            end
         end
      end
   end

   bestfs = best.features;

end

%{
function auc = revevalfs(part, features, nFold, method)
   folds = revgenfolds(part.X,part.Y,nFold);
   
   scores = []; Y = [];
   
   for i=1:nFold
      test = folds{i}.test;
      train = folds{i}.train;
      
      fmap = false(1,size(train.X,2)); fmap(features) = true;
      fmap(~activeset(train.X)) = false;
      train.X = train.X(:,fmap); test.X = test.X(:,fmap);
      
      if isempty(train.X)
         scores = [scores ; zeros(size(test.Y,1),1)];
         Y = [Y ; test.Y];
         continue;
      end
      
      %% Train SVM
      try
         warning off AMS:MaxIter
         [sig,C,alpha,b] = ams(train.X,train.Y, ...
            method,0,struct('itermax', 10000, 'nonorm', 1));
         warning on AMS:MaxIter
      catch ME
         auc = 0.0;
         return;
      end
      
      %% Test SVM
      Kt = compute_kernel(test.X, train.X, sig);
      scores = [scores ; (Kt*alpha)+b];
      Y = [Y ; test.Y];
      
   end
   
   auc = calcauc(scores,Y);
end
%}

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



function auc = revevalfs(part,fold,features,N,method)
   sHoldout = uint32(ceil(size(fold.train.Y,1) * 0.1));
   
   scores = []; Y = [];
   
   
   
   for f=1:N
      %% Build test
      L = size(fold.train.X,1);
      
      p = randperm(L);
      is = p(1:sHoldout);
      test = struct('X',fold.train.X(is,:),'Y',fold.train.Y(is),'ids',fold.train.ids(is));
      
      %% Build mutex training set
      keep = ~ismember(part.ids,test.ids);%true if not in test set
      train = struct('X', part.X(keep,:), 'Y', part.Y(keep), 'ids',part.ids(keep));
      assert(isempty(intersect(train.ids,test.ids)));
      
      %% Remove inactive features
      fmap = false(1,size(train.X,2)); fmap(features) = true;
      fmap(~activeset(train.X)) = false;
      train.X = train.X(:,fmap); test.X = test.X(:,fmap);
      
      if isempty(train.X)
         scores = [scores ; zeros(size(test.Y,1),1)];
         Y = [Y ; test.Y]
         continue;
      end
      
      %% Train SVM
      try
         warning off AMS:MaxIter
         [sig,C,alpha,b] = ams(train.X,train.Y, ...
            method,0,struct('itermax', 10000, 'nonorm', 1));
         warning on AMS:MaxIter
      catch ME
         auc = 0.0;
         return;
      end
      
      %% Test SVM
      Kt = compute_kernel(test.X, train.X, sig);
      
      ntscores = (Kt*alpha)+b;
      
      scores = [scores ; ntscores];
      Y = [Y ; test.Y];
      %sauc = sauc + calcauc(scores,test.Y);
   end
   
   auc = calcauc(scores,Y);
end


function active = activeset(X)
   L = size(X,1);
   tt = repmat(X(1,:),L,1);
   md = X - tt;
   active = any(md,1);
end

function [ tunes ] = revtune( folds,models,method)
   %REVTUNE Summary of this function goes here
   %   Detailed explanation goes here
   
   
   nFold = size(folds,1);
   
   tunes = cell(nFold,1);
   
   for i=1:nFold
      nPart = 5;
      
      ftunes = cell(nPart,1);
      
      for p=1:nPart
         
         model = models{i}(p);
         fold = folds{i};
         
         [scores,Y] = revscore(model,fold,nFold,method);
         [A , B] = revplatt(Y,scores);
         
         ftunes{p} = struct('A',A, 'B',B);
         
         fprintf('tuned fold %d/part %d A=%f B=%f\n',i,p,A, B);
         
      end
      tunes{i} = ftunes;
   end
   
end



function [scores,Y] = revscore(model,fold,N,method)
   sHoldout = uint32(ceil(size(fold.train.Y,1) * 0.1));
   
   part = model.parts;
   
   scores = []; Y = [];
   
   for f=1:N
      %% Build test
      p = randperm(sHoldout);
      test = struct('X',fold.train.X(p,:),'Y',fold.train.Y(p),'ids',fold.train.ids(p));
      
      %% Build mutex training set
      keep = ~ismember(part.ids,test.ids);%true if not in test set
      train = struct('X', part.X(keep,:), 'Y', part.Y(keep), 'ids',part.ids(keep));
      assert(isempty(intersect(train.ids,test.ids)));
      
      %% Remove inactive features
      fmap = false(1,size(train.X,2)); fmap(model.features) = true;
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
         scores = [scores ; zeros(size(test.Y,1),1)];
         Y = [Y ; test.Y];
         continue;
      end
      
      %% Test SVM
      Kt = compute_kernel(test.X, train.X, sig);
      scores = [scores ; (Kt*alpha)+b];
      Y = [Y ; test.Y];
      %sauc = sauc + calcauc(scores,test.Y);
   end
end

function active = activeset(X)
   L = size(X,1);
   tt = repmat(X(1,:),L,1);
   md = X - tt;
   active = any(md,1);
end

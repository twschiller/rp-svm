function [ auc ] = revtest( folds, models, method, tunes )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    result = struct('Y',[], 'scores',[]);

    nFold = size(folds,1);
    
    for i=1:nFold
       nPart = size(models{i},1);
      
       escores = zeros(size(folds{i}.test.Y,1),nPart);
       
       for j=1:nPart
          
          mff = models{i};
          model = mff(j);
          
          trainX = model.parts.X; trainY = model.parts.Y;
          testX = folds{i}.test.X;
          
          selected = false(1,size(trainX,2));
          selected(model.features) = true;
          
          active = markactive(trainX);
          
          trainX = trainX(:,active & selected);
          testX = testX(:,active & selected);
          
          %% Train SVM
          try
             warning off AMS:MaxIter
             [sig,C,alpha,b] = ams(trainX, trainY, ...
                method,0, struct('itermax', 10000, 'nonorm', 1));
             warning on AMS:MaxIter
          catch ME
             save('coredump.out');
             rethrow(ME);
          end
          Kt = compute_kernel(testX, trainX, sig);
          escores(:,j) = (Kt*alpha)+b;
          
          if (exist('tunes','var'))
             escores(:,j) = 1 ./ (1 + exp(tunes{i}{j}.A * escores(:,j)  + tunes{i}{j}.B));
          end
          
          innerauc = calcauc(folds{i}.test.Y,escores(:,j));
          
          fprintf('innerauc:%f\n',innerauc);
          
          clear sig C alpha b Kt
          
       end
        
       fscores = mean(escores,2);
       
       result.Y = [result.Y ; folds{i}.test.Y];
       result.scores = [result.scores ; fscores];
    end
    
    auc = calcauc(result.Y, result.scores);
    
end

function active = markactive(X)
    L = size(X,1);
    tt = repmat(X(1,:),L,1);
    md = X - tt;
    active = any(md,1);
end


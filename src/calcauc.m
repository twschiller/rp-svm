%% Calculate auc for all the scores offered
function auc = calcauc(Y, scores)
    %Calculate AUC using Hand's method from A Simple Generalisation of the Area 
    %Under the ROC Curve for Multiple Class Classification Problems (2001)
    %http://www.springerlink.com/content/nn141j42838n7u21/

    % AUC = [ S0 - n0(n0-1) / 2 ] / n0n1
    % Where n0 is the number of positive instances
    % n1 is the number negative instances
    % S0 = sum r_i , where r_i is the rank of the ith positive example in the
    % ranked list

    %% Argument validation
    error(nargchk(2, 2, nargin));
    assert(size(Y,1) == size(scores,1));
    assert(isvector(Y));
    assert(isvector(scores));
    
    %% Hand's method
    n0 = size(scores(Y > 0),1); %positive instances
    n1 = size(scores(Y < 0),1); %negative instances
    
    R = tiedrank(scores);
    S0 =  sum(R(Y > 0));
    
    num = S0 - ((n0 * (n0 + 1)) / 2);
    auc =  num / (n0 * n1);
end

function [ A , B ] = revplatt( Y , scores )
% Fit a sigmoid function to the SVM output

    %% Perform argument validation
    error(nargchk(2, 2, nargin));   
    error(nargchk(2, 2, nargout)); 
    
    %% Calculate the number of elements in stuff
    L = size(Y,1); %Number of test instances
    
    %% Calculate average scores        
    gscores = scores; 
        
    %% Build sigmoid fitting set and perform minimization
    np = sum(Y > 0);
    nn = L - np;
    
    [A,B] = cow(scores,Y,np,nn);

end

%% Fitness function
function [A,B] = cow(deci,label,prior1,prior0)
   maxiter=100; %Maximum number of iterations
   minstep=1e-10; %Minimum step taken in line search
   sigma=1e-12; %Set to any value > 0

   %Construct initial values: target support in array t, initial function value in fval
   hiTarget=(prior1+1.0)/(prior1+2.0); loTarget=1/(prior0+2.0);
   len=prior1+prior0; % Total number of data
   
   t = zeros(len,1);
   t(label>0) = hiTarget;
   t(label<0) = loTarget;
  
   
   A=0.0; B=log((prior0+1.0)/(prior1+1.0)); fval=0.0;
   
   for i=1:len
      fApB=deci(i)*A+B;
      if (fApB >= 0)
         fval = fval +  t(i)*fApB+log(1+exp(-fApB));
      else
         fval = fval + (t(i)-1)*fApB+log(1+exp(fApB));
      end
   end
   
   for it=1:maxiter
      h11=sigma;h22=sigma; h21=0.0;g1=0.0;g2=0.0;
      for i=1:len
         fApB=deci(i)*A+B;
         if (fApB >= 0)
            p=exp(-fApB)/(1.0+exp(-fApB)); q=1.0/(1.0+exp(-fApB));
         else
            p=1.0/(1.0+exp(fApB)); q=exp(fApB)/(1.0+exp(fApB));
         end 
         d2=p*q;
         h11 = h11 + deci(i)*deci(i)*d2; h22 = h22+ d2; h21 = h21 + deci(i)*d2;
         d1=t(i)-p;
         g1 = g1 + deci(i)*d1; g2 = g2 + d1;
      end
      if (abs(g1)<1e-5 && abs(g2)<1e-5) %Stopping criteria
         break;
      end
      
      det=h11*h22-h21*h21;
      dA=-(h22*g1-h21*g2)/det; dB=-(-h21*g1+h11*g2)/det;
      gd=g1*dA+g2*dB;
      stepsize=1;
      
      while (stepsize >= minstep) %//Line search
         newA=A+stepsize*dA; newB=B+stepsize*dB; newf=0.0;
         for i=1:len 
            fApB=deci(i)*newA+newB;
            if (fApB >= 0)
               newf = newf + t(i)*fApB+log(1+exp(-fApB));
            else
               newf = newf + (t(i)-1)*fApB+log(1+exp(fApB));
            end
         end
         if (newf<fval+0.0001*stepsize*gd)
            A=newA; B=newB; fval=newf;
            break; %Sufficient decrease satisfied
         else
            stepsize = stepsize / 2.0;
         end
         
      end
      if (stepsize < minstep)
         error('Platt:Line','Line search failed');
      end
   end
   if (it >= maxiter)
      fprintf('Reaching maximum iterations');
   end
end
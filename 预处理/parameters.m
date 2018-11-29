% This function set the parameters for the EMOTIC, CVPR 2017
% output,
% params: structure, parameters both defined by default
% 
function params = parameters
% Parameters of the loss fuction
% Prior Probability of each label
Pr = [8;6;5;27;50;23;26;11;26;2;5;3;6;3;1;4;2;1;2;1;2;2;3;1;1;2]*1e-2;
c = 1.2;
w =1./log(c+Pr);
lamda = [1/6 1];
theta = 0.5;
params = struct('Pr',Pr,'c',c,'w',w,'lamda',lamda,'theta',theta);
end


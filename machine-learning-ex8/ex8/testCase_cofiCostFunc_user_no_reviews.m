% -----------
% Test 6b (a user with no reviews):
% input:
params = [ 1:14 ] / 10;
Y = magic(4);
Y = Y(:,1:3);
R = [1 0 1; 1 1 1; 0 0 0; 1 1 0] > 0.5;     % R is logical
num_users = 3;
num_movies = 4;
num_features = 2;
lambda = 6;
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

% output:
% J =  331.08

% grad =
  % -15.5880
  % -22.3440
	% 1.8000
  % -12.5720
  % -18.4380
  % -26.8620
	% 4.2000
  % -14.7440
	% 1.9770
   % -1.0280
	% 4.5930
   % -5.0590
   % -8.2600
	% 1.9410

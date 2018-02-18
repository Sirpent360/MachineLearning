function [idx, Distance] = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);	%	Number of centroids

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m=size(X,1);	%	Calculating m - number of training examples

Distance=zeros(m, K);	%	Create a "distance" matrix of size (m x K) and initialize it to all zeros. 
						%	'm' is the number of training examples, 
						%	K is the number of centroids

for j=1:K				%	Use a for-loop over the 1:K centroids.
	D=bsxfun(@minus, X, centroids(j,:));	%	Inside this loop, create a column vector of the distance from each training example to that centroid, 
											%	and store it as a column of the distance matrix. 
											% One method is to use the bsxfun() function and the sum() function to calculate the sum of the squares of the differences 
											% between each row in the X matrix and a centroid.
	Distance(:,j)=sum(D.^2,2);	%	column vector of distance from each training example to certain centroid
end
	[d, d_idx]=min(Distance');	%	(')-transpose, because min works over columns(?), (we need min for one example between different centroids distances)
	idx=d_idx';					%	transpose, because we need column vector, not row vector

% =============================================================

end


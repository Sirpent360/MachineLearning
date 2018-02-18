for i = 1:m
    yVec(i,y(i)) = 1;
end


same as


y_matrix = eye(num_labels)(y,:) % 1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). This is most easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y'. 
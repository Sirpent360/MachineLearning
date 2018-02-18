y=[1; 2; 3; 4; 5; 6; 7; 8; 9; 10]
num_labels = 10

for t=1:10
    yy = (1:num_labels==y(t))'
end
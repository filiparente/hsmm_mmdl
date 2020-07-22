function matrix = generate_random_matrix(size, mode)

    matrix = zeros(size);
    
    for i=1:size(1)
        row = randsample(100, size(2));
        if strcmp(mode,'normal')==1
            row = row/sum(row);
        end

        matrix(i, :) = row;
    end
    if strcmp(mode,'zero-diag')==1
            for i=1:size(1)
                matrix(i,i) = 0.0;
                matrix(i,:) = matrix(i,:)/sum(matrix(i,:));
            end
    end
            
  
end
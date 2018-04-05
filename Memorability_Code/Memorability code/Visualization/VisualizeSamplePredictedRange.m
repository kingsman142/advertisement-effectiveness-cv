%% Visualize sample images over range

function [] = VisualizeSamplePredictedRange(predicted, test_label, test_set_img)

    [foo x] = sortrows([predicted test_label],-1);

    j = 0;
    for i=33:floor(size(x,1)/32):size(x,1)

        j = j+1;
        subplot(4,8,j);

        imshow(test_set_img(:,:,:,x(i)));
        title(sprintf('prediction: %g\nactual: %g', predicted(x(i)), test_label(x(i))));

    end
end


%% Noramlize image features

function [ normalized_image_features ] = NormalizeImageFeatures(image_features)

    normalized_image_features = image_features;
    for i = 1:size(image_features,2)
        colsum = sum(abs(image_features(:, i)));
        normalized_image_features(:, i) = image_features(:, i)*1/colsum;
    end
end


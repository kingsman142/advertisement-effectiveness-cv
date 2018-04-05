%% Visualize object-based memorability map

function [] = VisualizeObjectsMemorabilityMap(segments, object_memorabilities, scale_pos, scale_neg)

    segments_weighted = reshape(object_memorabilities(segments(:)+1),256,256);
    
    pos_image = (segments_weighted.*(segments_weighted>=0)-scale_pos(1))./scale_pos(2);
    neg_image = -(segments_weighted.*(segments_weighted<0)-scale_neg(1))./scale_neg(2);
    
    imshow(cat(3,pos_image+neg_image./4, (pos_image+neg_image)./4, neg_image+pos_image./4));
    axis('square');

end
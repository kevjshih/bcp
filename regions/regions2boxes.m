function regions = regions2boxes(region_list, superpixels)

   sp_areas_struct = regionprops(superpixels, 'BoundingBox');
   sp_bb = cat(1,sp_areas_struct.BoundingBox);
   sp_bb = [sp_bb(:,1) sp_bb(:,2) (sp_bb(:,1) + sp_bb(:,3)) (sp_bb(:,2) + sp_bb(:,4))];
   
   regions = get_region_bbox(region_list, sp_bb);

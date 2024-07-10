clear all
close all
file_directory_in = ".\Input_data\";
file_directory_out = ".\Output_data\";
load(file_directory_in+'measurements_YFP');
nutlinconcentrations = [2 1 0.5 0.250 0.125];
%%
for chosen_group=1:5 %Options: 1,2,3,4,5
    
    traces_YFP = measurements_YFP.filledSingleCellTraces;
    annotation = measurements_YFP.cellAnnotation;
    divisions = measurements_YFP.filledDivisionMatrixDataset;
    traces_YFP =traces_YFP(:,1:361);
    
    % check cells that have only one (or five) time point that is bad and extrapolate
    validCells = sum(traces_YFP == -1,2)==0;
    traces_YFP = traces_YFP(validCells,:);
    annotation = annotation(validCells,:);
    divisions = divisions(validCells,:);
    uniqueGroups = unique(annotation(:,1));
    [~, group_number] = ismember(annotation(:,1), uniqueGroups);
    
    %% select subset of cells for specific nutlin concentration
    subCells = group_number == chosen_group;
    traces_YFP=traces_YFP(subCells , :)./traces_YFP(subCells ,1);
    x_perturb_on=188; % when nutlin added, 
    x_perturb_off = 192; % washed at time point 192
    point_num=350; % for old data 277
    
    % Write data to text file
    writematrix(traces_YFP, file_directory_out+"p53_from_measurements_YFP_at"+num2str(nutlinconcentrations(chosen_group))+"uM.txt");
 end

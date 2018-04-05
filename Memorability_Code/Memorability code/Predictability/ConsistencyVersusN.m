%% Consistency versus number of subjects


%% consistency analysis
N_splits = 25;

clearvars subject_hrs1 subject_hrs2;
mean_scores_per_im = [];
rs = [];
all_rs = [];
total_scores = [];
N_steps = 25;
for j=1:N_steps
    
    j
    
    subj_prop = j/N_steps;
    % half splits
    N_images = 2400;

    subject_hits1 = zeros(N_splits, N_images);
    subject_hits2 = zeros(N_splits, N_images);
    subject_misses1 = zeros(N_splits, N_images);
    subject_misses2 = zeros(N_splits, N_images);
    for i=1:N_splits
        i
        [subject_hits1(i,:), subject_misses1(i,:), subject_hits2(i,:), subject_misses2(i,:)] = SplitHalfSubjectHitsMissesProportionSubjects(subj_data, subj_prop);
        
        subject_hrs1{i} = subject_hits1(i,:)./(subject_hits1(i,:)+subject_misses1(i,:));
        subject_hrs2{i} = subject_hits2(i,:)./(subject_hits2(i,:)+subject_misses2(i,:));
        
        total_scores(i,j) = sum(subject_hits1(i,:) + subject_misses1(i,:) + subject_hits2(i,:) + subject_misses2(i,:));
    end
    
    %total_scores = sum(subject_hits1(:) + subject_misses1(:) + subject_hits2(:) + subject_misses2(:));
    mean_scores_per_im(j) = sum(total_scores(:,j))/(N_splits*N_images);
    
    
    use_all_images_human_consistency = 't';
    
    [r, rho, mse, all_r, all_rho, all_mse] = GetConsistencyStats(subject_hrs1, subject_hrs2, use_all_images_human_consistency, []);
    
    rs = [rs r];
    all_rs = [all_rs all_r];
end


%%
x = mean_scores_per_im;
%x = ((1:N_steps)/N_steps)*length(subj_data);
y = rs;
scatter(x,y,'.','blue');
hold on;
xlabel('Mean number of scores per image');
%xlabel('Number of subjects in experiment');
ylabel('r');
%title('Consistency versus number of subjects in experiment');
title('Consistency versus mean number of scores per image');
set(gca,'xlim',[0,80]);
set(gca,'ylim',[0,1]);
%p = polyfit(x,35.^y,1);
%x2 = ((0:N_steps)/N_steps)*100;
%f = polyval(p,x2);
%plot(x2,logb(f,35),'black');
%plot([1,80],[0.8,0.8],'--','Color',[0.5,0.5,0.5]);
plot([1,80],[max(rs),max(rs)],'--','Color',[0.5,0.5,0.5]);
%plot([29,29],[0,1],'--','Color',[0.5,0.5,0.5]);

e = Get80PercentIntervals2(all_rs);
errorbar(x, y, abs(e(:,1)), e(:,2), '.');

pbaspect([0.4 1 1]);
set(gca,'xtick',0:20:80);


%%
print('-dpsc', fullfile('../Analysis/plots/',which_date,'consistency_vs_n_scores_per_image.eps'));
close all;



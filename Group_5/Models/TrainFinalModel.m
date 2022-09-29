close all
clear all
rng(4599136)
%rng(18061998)

%load('June2017.mat')
%load('June2017labels.mat')
load('ALL.mat')
load('ALLlabels')
load('Dopplerf.mat')

%%

centroid = nan(size(CompleteMat,1),size(CompleteMat,3));
BW = nan(size(CompleteMat,1),size(CompleteMat,3));

for k = 1 : size(CompleteMat,1)
k
% F = 20*log10(abs(squeeze(CompleteMat(k,:,:))));
F = squeeze(CompleteMat(k,:,:));


for j = 1:size(F,2)
    Fvec = F(:,j);%F(:,j);
    f = Dopplerf';
    centroid(k,j) = sum(f.*Fvec)/sum(Fvec);
    %BW(k,j) = sqrt( (sum(((f-centroid(k,j)).^2) .*Fvec)));%/(sum(Fvec)) );
    %BW(k,j) = sqrt( (sum(((f - centroid(k,j)).^2) .*Fvec))/(sum(Fvec)) );
    BW(k,j) = sqrt( (sum(((f - centroid(k,j)).^2) .*Fvec))/(sum(sum(F))));
end

end

%% 
segs = 10%segs=5 %Split in 10 segments
featuremat = nan(size(CompleteMat,1),8*segs);
for k = 1 :  size(CompleteMat,1)
    startidx = 1;
    meanc = nan(1,segs);
    meanBW = nan(1,segs);
    varc = nan(1,segs);
    varBW = nan(1,segs);
    skewc = nan(1,segs);
    skewBW = nan(1,segs);
    kurtc = nan(1,segs);
    kurtBW = nan(1,segs);
    for i = 1:segs
    endidx = round(size(centroid(k,:),2)*i/segs);

    cseg = centroid(k,startidx:endidx);
    BWseg = BW(k,startidx:endidx);

    meanc(i) = mean(cseg);
    meanBW(i) = mean(BWseg);
    varc(i) = var(cseg);
    varBW(i) = var(BWseg);
    skewc(i) = skewness(cseg);
    skewBW(i) = skewness(BWseg);
    kurtc(i) = kurtosis(cseg);
    kurtBW(i) = kurtosis(BWseg);

    startidx = endidx+1;
    end
    featuremat(k,:) = [meanc, meanBW, varc, varBW, skewc, skewBW, kurtc, kurtBW]; %needs to become 162x80
end


%% Bop code
CompleteMat = flip(CompleteMat,2);
labels(length(labels)+1:1:2*length(labels)) = labels;

%%

centroid = nan(size(CompleteMat,1),size(CompleteMat,3));
BW = nan(size(CompleteMat,1),size(CompleteMat,3));

for k = 1 : size(CompleteMat,1)
k
% F = 20*log10(abs(squeeze(CompleteMat(k,:,:))));
F = squeeze(CompleteMat(k,:,:));


for j = 1:size(F,2)
    Fvec = F(:,j);%F(:,j);
    f = Dopplerf';
    centroid(k,j) = sum(f.*Fvec)/sum(Fvec);
    %BW(k,j) = sqrt( (sum(((f-centroid(k,j)).^2) .*Fvec)));%/(sum(Fvec)) );
    %BW(k,j) = sqrt( (sum(((f - centroid(k,j)).^2) .*Fvec))/(sum(Fvec)) );
    BW(k,j) = sqrt( (sum(((f - centroid(k,j)).^2) .*Fvec))/(sum(sum(F))));
end

end

%% 
featuremat2 = nan(size(CompleteMat,1),8*segs);
for k = 1 :  size(CompleteMat,1)
    startidx = 1;
    meanc = nan(1,segs);
    meanBW = nan(1,segs);
    varc = nan(1,segs);
    varBW = nan(1,segs);
    skewc = nan(1,segs);
    skewBW = nan(1,segs);
    kurtc = nan(1,segs);
    kurtBW = nan(1,segs);
    for i = 1:segs
    endidx = round(size(centroid(k,:),2)*i/segs);

    cseg = centroid(k,startidx:endidx);
    BWseg = BW(k,startidx:endidx);

    meanc(i) = mean(cseg);
    meanBW(i) = mean(BWseg);
    varc(i) = var(cseg);
    varBW(i) = var(BWseg);
    skewc(i) = skewness(cseg);
    skewBW(i) = skewness(BWseg);
    kurtc(i) = kurtosis(cseg);
    kurtBW(i) = kurtosis(BWseg);

    startidx = endidx+1;
    end
    featuremat2(k,:) = [meanc, meanBW, varc, varBW, skewc, skewBW, kurtc, kurtBW]; %needs to become 162x80
end

%% Merge data with augmented data

featuremat = [featuremat ; featuremat2];

%% train the final pipeline model
idx = randsample(length(labels),length(labels));

X = featuremat(idx,:);
X = (featuremat(idx,:)- repmat(mean(featuremat,1),length(labels),1))./(repmat(std(featuremat,1),length(labels),1));
Y = labels(idx);

t = templateTree('Reproducible',true);
finalModel = fitcensemble(X,Y,'OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
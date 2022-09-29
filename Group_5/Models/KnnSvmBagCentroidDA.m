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

%% Test with KNN
idx = randsample(length(labels),length(labels));

X = featuremat(idx,:);
X = (featuremat(idx,:)- repmat(mean(featuremat,1),length(labels),1))./(repmat(std(featuremat,1),length(labels),1));
Y = labels(idx);

split = 0.85; %train test split
trainidx = round(split*length(labels));

%simple holdout
Xtrain = X(1:trainidx,:);
Xtest = X(trainidx+1:end,:);
Ytrain = Y(1:trainidx);
Ytest = Y(trainidx+1:end);
for i =1:50
    MyModel1 = fitcknn(Xtrain,Ytrain);
    MyModel1.NumNeighbors = i;
    MyPredictedLabels=predict(MyModel1,Xtest);
    MyCM1 = confusionmat(Ytest, MyPredictedLabels);
    acc(i) = sum(Ytest==MyPredictedLabels)/length(MyPredictedLabels);
end
figure
plot(acc)
acc_noFeatureSelection = acc(7)


%% Feature selection
fun = @(XT,yT,Xt,yt)KNNseq(XT,yT,Xt,yt);
c = cvpartition(Ytrain,'k',10);
opts = statset('Display','iter');
[inmodel,history] = sequentialfs(fun,Xtrain,Ytrain,'cv',c,'options',opts,'direction','forward')


Xtrainselected = Xtrain(:,inmodel);
ytrainselected = Ytrain;
Xtestselected = Xtest(:,inmodel);
ytestselected = Ytest;

Modelsel = fitcknn(Xtrainselected,ytrainselected);
Modelsel.NumNeighbors = 7;
MyPredictedLabels=predict(Modelsel,Xtestselected);
selCM = confusionmat(ytestselected, MyPredictedLabels);
acc = sum(ytestselected==MyPredictedLabels)/length(MyPredictedLabels);


%% Test with SVM and 
MySVMsel = fitcecoc(Xtrainselected,ytrainselected);

t = templateSVM('Standardize',true,'KernelFunction','gaussian');
MySVMsel = fitcecoc(Xtrainselected,ytrainselected,'Learners',t,'FitPosterior',true,...
        'Verbose',2);

MyPredictedLabelsSVMsel=predict(MySVMsel,Xtestselected);
accSVMsel = sum(ytestselected==MyPredictedLabelsSVMsel)/length(MyPredictedLabelsSVMsel);



MySVM = fitcecoc(Xtrain,Ytrain);
t = templateSVM('Standardize',true,'KernelFunction','gaussian');
% MySVM = fitcecoc(Xtrain,Ytrain,'Learners',t,'FitPosterior',true,...
%         'Verbose',2);
    
MyPredictedLabelsSVM=predict(MySVM,Xtest);
accSVM = sum(Ytest==MyPredictedLabelsSVM)/length(MyPredictedLabelsSVM);


%% Ensemble with selected features
t = templateTree('Reproducible',true);
Mdl = fitcensemble(Xtrainselected,ytrainselected,'OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%%% Predict ensamble
Enslabels=predict(Mdl,Xtestselected);
EnsCMsel = confusionmat(ytestselected, Enslabels);
EnsAccSel = sum(ytestselected==Enslabels)/length(Enslabels);

%% Classification ensemble with all features

t = templateTree('Reproducible',true);
Mdl = fitcensemble(Xtrain,Ytrain,'OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%%% Predict ensamble
Enslabels=predict(Mdl,Xtest);
EnsCM = confusionmat(Ytest, Enslabels);
EnsAcc = sum(Ytest==Enslabels)/length(Enslabels);

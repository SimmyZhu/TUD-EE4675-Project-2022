clear all
close all
rng(4599136)

load('ALL.mat')
load('ALLlabels.mat')
load('Dopplerf.mat')
%%
orderMax = 8;

featureMat = nan(length(labels),(orderMax+1)^2);

testspecto = squeeze(CompleteMat(1,:,:));
testCVD = fft(abs(testspecto),[],2);

L = size(testCVD,2); %Determine length of image => 481
H = size(testCVD,1); %Determine heigth of image => 800
rangeH = linspace(1,H,H); %Get vector of indeces for chebychev polynomials
rangeL = linspace(1,L,L); %Get vector of indeces for chebychev polynomials

Chebmatfreq = nan(orderMax+1,H);
Chebmatcad = nan(orderMax+1,L);
for i = 0:orderMax
    Chebmatfreq(i+1,:) = chebyshevT(i,rangeH);
    Chebmatcad(i+1,:) = chebyshevT(i,rangeL);
end

for i = 1: length(labels)
i

testspecto = squeeze(CompleteMat(i,:,:));
testCVD = fft(abs(testspecto),[],2);


feature = nan(orderMax+1, orderMax+1);
for h=0:orderMax
    polyfreq = Chebmatfreq(h+1,:);        %Get chebycehv polynomials 
    polyfreq = polyfreq/(H^h);              %Normalise with beta
    for l=0:orderMax
        polycad =  Chebmatcad(l+1,:);    %Get chebycehv polynomials 
        polycad = polycad/(L^l);            %Normalise with beta
        polymat = polyfreq' * polycad;      %Determine all crossproducts of the polynomials
        feature(h+1,l+1) = sum(sum(polymat.*abs(testCVD))); %Sum over all crossproducts multiplied with CVD
        %rholL = factorial(2*l)* nchoosek(L+l,2*l+1);
        %rhohH = factorial(2*h)* nchoosek(H+h,2*h+1);
        %feature(h+1,l+1) = feature(h+1,l+1)/(rholL*rhohH);        %Normalise with rho
    end
end
feature = reshape(feature,1,[]);
feature = (feature - mean(feature))/std(feature);

featureMat(i,:) = feature;
end





%% Bop code
CompleteMat = flip(CompleteMat,2);
labels(length(labels)+1:1:2*length(labels)) = labels;

%% Augmented cheby
featureMat2 = nan(length(labels)/2,(orderMax+1)^2);

for i = 1: length(labels)/2
i

testspecto = squeeze(CompleteMat(i,:,:));
testCVD = fft(abs(testspecto),[],2);


feature = nan(orderMax+1, orderMax+1);
for h=0:orderMax
    polyfreq = Chebmatfreq(h+1,:);        %Get chebycehv polynomials 
    polyfreq = polyfreq/(H^h);              %Normalise with beta
    for l=0:orderMax
        polycad =  Chebmatcad(l+1,:);    %Get chebycehv polynomials 
        polycad = polycad/(L^l);            %Normalise with beta
        polymat = polyfreq' * polycad;      %Determine all crossproducts of the polynomials
        feature(h+1,l+1) = sum(sum(polymat.*abs(testCVD))); %Sum over all crossproducts multiplied with CVD
        %rholL = factorial(2*l)* nchoosek(L+l,2*l+1);
        %rhohH = factorial(2*h)* nchoosek(H+h,2*h+1);
        %feature(h+1,l+1) = feature(h+1,l+1)/(rholL*rhohH);        %Normalise with rho
    end
end
feature = reshape(feature,1,[]);
feature = (feature - mean(feature))/std(feature);

featureMat2(i,:) = feature;
end


%% Merge with augmented data
featuremat = [featureMat ; featureMat2];


%% Test with KNN
idx = randsample(length(labels),length(labels));

%%
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


%% Classification ensemble

t = templateTree('Reproducible',true);
Mdl = fitcensemble(Xtrain,Ytrain,'OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%%% Predict ensamble
Enslabels=predict(Mdl,Xtest);
EnsCM = confusionmat(Ytest, Enslabels);
EnsAcc = sum(Ytest==Enslabels)/length(Enslabels);



%% adaBoost
% tTree = templateTree('MinLeafSize',20);
% t = templateEnsemble('AdaBoostM1',100,tTree,'LearnRate',0.1);
% Md2 = fitcensemble(Xtrain,Ytrain,t);
% Ens2labels=predict(Md2,Xtest);
% Ens2CM = confusionmat(Ytest, Ens2labels);
% Ens2Acc = sum(Ytest==Ens2labels)/length(Ens2labels);

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

%% Ensemble with selected features
t = templateTree('Reproducible',true);
Mdl = fitcensemble(Xtrainselected,ytrainselected,'OptimizeHyperparameters','auto','Learners',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%%% Predict ensamble
Enslabels=predict(Mdl,Xtestselected);
EnsCM = confusionmat(ytestselected, Enslabels);
EnsAccSel = sum(ytestselected==Enslabels)/length(Enslabels);

%% Fit svm on selected features
MySVMsel = fitcecoc(Xtrainselected,ytrainselected);

t = templateSVM('Standardize',true,'KernelFunction','gaussian');
MySVMsel = fitcecoc(Xtrainselected,ytrainselected,'Learners',t,'FitPosterior',true,...
        'Verbose',2);

MyPredictedLabelsSVMsel=predict(MySVMsel,Xtestselected);
accSVMsel = sum(ytestselected==MyPredictedLabelsSVMsel)/length(MyPredictedLabelsSVMsel);



MySVM = fitcecoc(Xtrain,Ytrain);
t = templateSVM('Standardize',true,'KernelFunction','gaussian');
%MySVM = fitcecoc(Xtrain,Ytrain,'Learners',t,'FitPosterior',true,...
%        'Verbose',2);
    
MyPredictedLabelsSVM=predict(MySVM,Xtest);
accSVM = sum(Ytest==MyPredictedLabelsSVM)/length(MyPredictedLabelsSVM);




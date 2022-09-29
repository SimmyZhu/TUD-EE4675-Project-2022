%Load the data (make sure the path is correct in the LoadDataPipelin script)
LoadDataPipeline
%Load the trained model
load('finalModel.mat')


%Create the features
load('Dopplerf.mat')
centroid = nan(size(CompleteMat,1),size(CompleteMat,3));
BW = nan(size(CompleteMat,1),size(CompleteMat,3));
for k = 1 : size(CompleteMat,1)
k
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

segs=10 %Split in 10 segments
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
    featuremat(k,:) = [meanc, meanBW, varc, varBW, skewc, skewBW, kurtc, kurtBW]; %needs to become 100x80
end


Xnorm = (featuremat- repmat(mean(featuremat,1),size(featuremat,1),1))./(repmat(std(featuremat,1),size(featuremat,1),1));


%classify the unseen cases
Predictions=predict(finalModel,Xnorm);

%Sort the cases
Cases = str2double(caseNum);
Cases(3) = 100; %hardcoded, readfile reads in wrong order
[Cases, idx] = sort(Cases);
Predictions2 = Predictions(idx);
casePredict = [Cases, Predictions2]; %Final matrix with cases and the predictions
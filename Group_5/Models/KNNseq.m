function [crit] = KNNseq(Xtrain,ytrain, Xtest, ytest)
%Function to use in forward feature selection, using KNN
    MyModel1 = fitcknn(Xtrain,ytrain);
    MyModel1.NumNeighbors = 7;
    MyPredictedLabels=predict(MyModel1,Xtest);
    acc = sum(ytest==MyPredictedLabels)/length(MyPredictedLabels);
    crit = -1*acc;
end


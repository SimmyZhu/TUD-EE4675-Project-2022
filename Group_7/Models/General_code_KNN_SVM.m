%% Creates a quick look of Range-Time and Spectrograms from a data file
%==========================================================================
% Author UoG Radar Group
% Version 1.0

% The user has to select manually the range bins over which the
% spectrograms are calculated. There may be different ways to calculate the
% spectrogram (e.g. coherent sum of range bins prior to STFT). 
% Note that the scripts have to be in the same folder where the data file
% is located, otherwise uigetfile() and textscan() give an error. The user
% may replace those functions with manual read to the file path of a
% specific data file
%==========================================================================

%% Decide if we need to create the feature matrix again or not

decide_feature_extraction = 1;

if decide_feature_extraction == 1

    %% Data reading part
    clear;

    filePattern = fullfile('*.dat');
    files = dir(filePattern);

    N_files = length(files);
%     N_files = 6;
    N_features = 8;

    %initialize some variables
    X = zeros(N_files,N_features);
    Y = zeros(N_files,1);


     for i_file = 1:N_files

    i_file
        Data_range = [];
        % [filename,pathname] = uigetfile('*.dat');
    %fileID = fopen(filename, 'r');
    fileID = fopen(files(i_file).name, 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);
    radarData = dataArray{1};
    clearvars fileID dataArray ans;
    fc = radarData(1); % Center frequency
    Tsweep = radarData(2); % Sweep time in ms
    Tsweep=Tsweep/1000; %then in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
    % For CW, it is 0.
    Data = radarData(5:end); % raw data in I+j*Q format
    fs=NTS/Tsweep; % sampling frequency ADC
    record_length=length(Data)/NTS*Tsweep; % length of recording in s
    nc=record_length/Tsweep; % number of chirps

    %% Reshape data into chirps and plot Range-Time
    Data_time=reshape(Data, [NTS nc]);
    win = ones(NTS,size(Data_time,2));
    %Part taken from Ancortek code for FFT and IIR filtering
    tmp = fftshift(fft(Data_time.*win),1);
    Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);
    ns = oddnumber(size(Data_range,2))-1;
    Data_range_MTI = zeros(size(Data_range,1),ns);
    [b,a] = butter(4, 0.0075, 'high');
    [h, f1] = freqz(b, a, ns);
    for k=1:size(Data_range,1)
      Data_range_MTI(k,1:ns) = filter(b,a,Data_range(k,1:ns));
    end
    freq =(0:ns-1)*fs/(2*ns); 
    range_axis=(freq*3e8*Tsweep)/(2*Bw);
    Data_range_MTI=Data_range_MTI(2:size(Data_range_MTI,1),:);
    Data_range=Data_range(2:size(Data_range,1),:);
    % figure
    % colormap(jet)
    % % imagesc([1:10000],range_axis,20*log10(abs(Data_range_MTI)))
    % imagesc(20*log10(abs(Data_range_MTI)))
    % % imagesc(20*log10(abs(Data_range)))
    % xlabel('No. of Sweeps')
    % ylabel('Range bins')
    % title('Range Profiles after MTI filter')
    % clim = get(gca,'CLim'); axis xy; ylim([1 100])
    % set(gca, 'CLim', clim(2)+[-60,0]);
    % drawnow

    %% Spectrogram processing for 2nd FFT to get Doppler
    % This selects the range bins where we want to calculate the spectrogram
    % bin_indl = 10
    % bin_indu = 30

    bin_indl = 5;
    bin_indu = 60;

    MD.PRF=1/Tsweep;
    MD.TimeWindowLength = 200;
    MD.OverlapFactor = 0.95;
    MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
    MD.Pad_Factor = 4;
    MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
    MD.DopplerBin=MD.PRF/(MD.FFTPoints);
    MD.DopplerAxis=-MD.PRF/2:MD.DopplerBin:MD.PRF/2-MD.DopplerBin;
    MD.WholeDuration=size(Data_range_MTI,2)/MD.PRF;
    MD.NumSegments=floor((size(Data_range_MTI,2)-MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

    Data_spec_MTI2=0;
    Data_spec2=0;
    for RBin=bin_indl:1:bin_indu
        Data_MTI_temp = fftshift(spectrogram(Data_range_MTI(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
        Data_spec_MTI2=Data_spec_MTI2+abs(Data_MTI_temp);                                
        Data_temp = fftshift(spectrogram(Data_range(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
        Data_spec2=Data_spec2+abs(Data_temp);
    end
    MD.TimeAxis=linspace(0,MD.WholeDuration,size(Data_spec_MTI2,2));

    Data_spec_MTI2=flipud(Data_spec_MTI2);

    %Ways of normalizing the Data_spec_MTI2
    %Maybe we shouldn't do anything
%     Data_spec_MTI2 = Data_spec_MTI2./max(max(Data_spec_MTI2));
    Data_spec_MTI2 = Data_spec_MTI2./mean(mean(Data_spec_MTI2));

    figure
     imagesc(MD.TimeAxis,MD.DopplerAxis.*3e8/2/5.8e9,20*log10(abs(Data_spec_MTI2))); colormap('jet'); axis xy
    % imagesc(MD.TimeAxis,MD.DopplerAxis.*3e8/2/5.8e9,20*log10(abs(Data_spec2))); colormap('jet'); axis xy
    ylim([-5 5]); colorbar
    colormap; %xlim([1 9])
    clim = get(gca,'CLim');
    set(gca, 'CLim', clim(2)+[-40,0]);
    xlabel('Time[s]', 'FontSize',16);
    ylabel('Velocity [m/s]','FontSize',16)
    set(gca, 'FontSize',16)
    title(files(i_file).name)

    %% Extracting the features

    %CENTROID
    
    doppler = MD.DopplerAxis;

    centroid = (doppler*Data_spec_MTI2)./(sum(Data_spec_MTI2));

%     figure
%     plot(MD.TimeAxis,centroid)

    cent_skew = skewness(centroid);
    cent_kurt = kurtosis(centroid);
    cent_mean = mean(centroid);
    cent_max = max(centroid);
    cent_min = min(centroid);

    %BANDWIDTH (Not using)
    
    N_doppler = size(Data_spec_MTI2,1);
    N_time = size(Data_spec_MTI2,2);
    % 
    % doppler_matrix = repmat(doppler,N_time,1);
    % cent_matrix = repmat(centroid,N_doppler,1)';
    % 
    % doppler_cent = doppler_matrix - cent_matrix;
    % doppler_cent = doppler_cent.^2;
    % numerator = sum(doppler_cent.*Data_spec_MTI2',2)';
    % B = numerator./(sum(Data_spec_MTI2));
    % B = B.^(1/2);

    % Not using the Bandwidth anymore
%     N_doppler2 = length(250:550);
%     N_time = size(Data_spec_MTI2,2);
% 
%     doppler_matrix = repmat(doppler(250:550),N_time,1);
%     cent_matrix = repmat(centroid,length(250:550),1)';
% 
%     doppler_cent = doppler_matrix - cent_matrix;
%     doppler_cent = doppler_cent.^2;
%     numerator = sum(doppler_cent.*Data_spec_MTI2(250:550,:)',2)';
%     B = numerator./(sum(Data_spec_MTI2(250:550,:)));
%     B = B.^(1/2);
% 
%     B_mean = mean(B);

%     figure
%     plot(MD.TimeAxis,B)

    %ENVELOPE
    env_up = zeros(1,N_time);
    env_down = zeros(1,N_time);
    
    for t = 1:N_time
       
        for dp = 1:N_doppler/2
           
            if Data_spec_MTI2(dp + N_doppler/2, t) > 4e5
                env_up(t) = MD.DopplerAxis(dp+N_doppler/2);
            end
            if Data_spec_MTI2(-dp + N_doppler/2 + 1, t) > 4e5
                env_down(t) = MD.DopplerAxis(-dp+N_doppler/2 +1);
            end
        end
    end
    
%     figure
%     plot(MD.TimeAxis,env_up,MD.TimeAxis,env_down);
    
    [env_max, i_env_max] = max(env_up);
    [env_min, i_env_min] = min(env_down);
    env_mean = mean(env_up - env_down);
    env_max_dist = env_up(i_env_max) - env_down(i_env_max);
    env_min_dist = -env_up(i_env_min) + env_down(i_env_min);


    split = textscan(files(i_file).name, '%f%s');

    X(i_file,:) = [cent_kurt cent_skew cent_mean env_mean env_max env_min env_max_dist env_min_dist];
    Y(i_file) = split{1};

     end
    
%% Standardization
    X_raw = X;
    X_iqr = normalize(X,'scale','iqr');
    X_n = normalize(X);
    
end
%% Machine Learning Part
X = X_n;

%% SVM Model


Scores = [];
classes=unique(Y);
ms=length(classes);
SVMModels=cell(ms,1);
for j = 1:numel(classes)
    indx=(Y==classes(j));%strcmp(Y,classes(j)); % Create binary classes for each classifier
    SVMModels{j}=fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','polynomial');
end

for j=1:numel(classes)
    [~,score]=predict(SVMModels{j},X);
    Scores(:,j)=score(:,2); % Second column contains positive-class scores
end
[~,maxScore]=max(Scores,[],2);
figure
confusionchart(Y, maxScore);
MyCM1 = confusionmat(Y, maxScore);

Mycp=cvpartition(Y, 'Kfold',3);
for j=1:numel(classes)
    MyCValModel = crossval(SVMModels{j}, 'CVpartition', Mycp);
    MyCVAccuracy(j) = 1-kfoldLoss(MyCValModel);
end

MyCVAccuracy

%% SVM Overall accuracy

K = 10; % The number of folds
your_data = X;
your_classes = Y;
N = size(your_data, 1); % The number of data samples to train / test

accuracy = [];
for i = 1:K
    
    cv = cvpartition(size(Y,1),'HoldOut',0.1);
    idx = cv.test;
    
    X_train = X(~idx, :); % The data to train on, 90% of the total.
    Y_train = Y(~idx, :); % The class labels of your training data.
    X_test = X(idx, :); % The data to test on, 10% of the total.
    Y_test = Y(idx, :); % The class labels of your test data.

    classes=unique(Y);
    ms=length(classes);
    SVMModels=cell(ms,1);
    for j = 1:numel(classes)
        indx=(Y_train==classes(j));%strcmp(Y,classes(j)); % Create binary classes for each classifier
        SVMModels{j}=fitcsvm(X_train,indx,'ClassNames',[false true],'Standardize',true,...
            'KernelFunction','polynomial');
    end
    Scores = [];
    for j=1:numel(classes)
        [~,score]=predict(SVMModels{j},X_test);
        Scores(:,j)=score(:,2); % Second column contains positive-class scores
    end
    [~,maxScore]=max(Scores,[],2);
    cont(i) = 0;
    for ind = 1: numel(maxScore)
       if maxScore(ind) == Y_test(ind)
           cont(i) = cont(i) + 1;
       end
    end
    accuracy(i) = cont(i)/numel(maxScore);
    accuracy(i);

end
overall_accuracy = mean(accuracy)

%% TSNE and KNN Model

X = X_n(:,3:6);

% Graphical visualization
figure
g_view = tsne(X);
gscatter(g_view(:,1),g_view(:,2),Y)
% 
% 
% MyModel1 = fitcknn(X,Y);
MyModel1 = fitcknn(X,Y,'NumNeighbors',4);
MyPredictedLabels=predict(MyModel1,X);
% figure
% confusionchart(Y, MyPredictedLabels);
% MyCM1 = confusionmat(Y, MyPredictedLabels);

Mycp=cvpartition(Y, 'Kfold',6);
MyCValModel = crossval(MyModel1, 'CVpartition', Mycp);
MyCVAccuracy = 1-kfoldLoss(MyCValModel)

MyPredictedLabels=predict(MyModel1,MyCValModel.X);
% 
figure
confusionchart(MyCValModel.Y, MyPredictedLabels);
MyCM1 = confusionmat(MyCValModel.Y, MyPredictedLabels);

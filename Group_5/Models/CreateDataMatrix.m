close all
clear all
rng(459936)

% Specify the folder where the files live.
myFolder = 'C:\Users\Wervers\Documents\school en studies\TU Delft\EE4675 Object Classification with Radar\Project\data\ALL\';
myFolder = 'C:\Users\Wervers\Documents\school en studies\TU Delft\EE4675 Object Classification with Radar\Project\data\Unseen\'
%myFolder = 'C:\Users\Wervers\Documents\school en studies\TU Delft\EE4675 Object Classification with Radar\Project\data\3 June 2017 Dataset\';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.dat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
CompleteMat = nan(length(theFiles),800,481);
labels = nan(length(theFiles),1);
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);

    fileID = fopen([myFolder, baseFileName], 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);
    radarData = dataArray{1};
    
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

    %Reshape data into chirps and plot Range-Time
    Data_time=reshape(Data, [NTS nc]);
    win = ones(NTS,size(Data_time,2));
    %Part taken from Ancortek code for FFT and IIR filtering
    tmp = fftshift(fft(Data_time.*win),1);
    
    clear Data_range
    Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);
    ns = oddnumber(size(Data_range,2))-1;
    Data_range_MTI = zeros(size(Data_range,1),ns);
    [b,a] = butter(4, 0.0075, 'high');
    [h, f1] = freqz(b, a, ns);
    for l=1:size(Data_range,1)
      Data_range_MTI(l,1:ns) = filter(b,a,Data_range(l,1:ns));
    end
    freq =(0:ns-1)*fs/(2*ns); 
    range_axis=(freq*3e8*Tsweep)/(2*Bw);
    Data_range_MTI=Data_range_MTI(2:size(Data_range_MTI,1),:);
    Data_range=Data_range(2:size(Data_range,1),:);
    
    
%    figure
%    colormap(jet)
%     imagesc([1:10000],range_axis,20*log10(abs(Data_range_MTI)))
%     imagesc(20*log10(abs(Data_range_MTI)))
%     xlabel('No. of Sweeps')
%     ylabel('Range bins')
%     title('Range Profiles after MTI filter')
%     clim = get(gca,'CLim'); axis xy; ylim([1 100])
%     set(gca, 'CLim', clim(2)+[-60,0]);
%     drawnow

    bin_indl = 10;
    bin_indu = 30;

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
    
    if size(Data_spec_MTI2,2) > 481
        Data_spec_MTI2 = Data_spec_MTI2(:,end-480:end);
    end
    
    %sizes(k,:,:) = size(Data_spec_MTI2)
    %CompleteMat(k,:,:) = 20*log10(abs(Data_spec_MTI2));
    CompleteMat(k,:,:) = Data_spec_MTI2;
    labels(k) = str2double(baseFileName(1));
    
    
% figure
% imagesc(MD.TimeAxis,MD.DopplerAxis.*3e8/2/5.8e9,20*log10(abs(Data_spec_MTI2))); colormap('jet'); axis xy
% ylim([-6 6]); colorbar
% colormap; %xlim([1 9])
% clim = get(gca,'CLim');
% set(gca, 'CLim', clim(2)+[-40,0]);
% xlabel('Time[s]', 'FontSize',16);
% ylabel('Velocity [m/s]','FontSize',16)
% set(gca, 'FontSize',16)

end
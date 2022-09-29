%    
%   Bob van Nifterik- 4558421
%   Jurgen Wervers - 4599136
%   resize radar data and save as png
%   The created file structure crated determines the label 
%


% load  dataset
load('ALL.mat')
load('Dopplerf.mat')
load('ALLlabels.mat')



%%
%set main map path 
% be aware that the sub-folders representing the labels must already be present
DirectoryPath ='C:\Users\bobni\OneDrive\Documenten\Miniproject\imgData240\';
for i = 1:length(labels)
tempImg = squeeze(mat2gray(20*log10(abs(CompleteMat(i,:,:)))));
img = imresize(tempImg,  [224,224]);
imwrite(img, [DirectoryPath,num2str(labels(i)),'\sample',num2str(i),'.png'] )
i
end


%% flip data and calculate again - uncomment 
% CompleteMatFlip = flip(CompleteMat,2);

%%
DirectoryPath ='C:\Users\bobni\OneDrive\Documenten\Miniproject\imgDataCVD\';
for i = 1:length(labels)
%tempImg = squeeze(mat2gray(20*log10(abs(CompleteMat(i,:,:)))));


% for cvd
F = squeeze(CompleteMatFlip(i,:,:));
CVD = (squeeze(fft(F,[],2))); 
tempImg = squeeze(mat2gray(20*log10(abs(CVD))));

%
img = imresize(tempImg,  [224,224]);
rgbImage = cat(3, img, img, img);
imwrite(rgbImage, [DirectoryPath,num2str(labels(i)),'\sample',num2str(i),'flip.png'] )
i
end

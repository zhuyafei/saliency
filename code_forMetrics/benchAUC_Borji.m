%% CVBIOUC
function benchAUC_Borji()
InputGroundTruth = './tmp/Datasets/GroundTruth/';
InputSaliencyMap = './tmp/SaliencyMaps/';
%OutputResultsROC = './tmp/Results/roc/';
OutputResultsAUC = './tmp/Results/aucBorji/';
traverse(InputGroundTruth, InputSaliencyMap, OutputResultsAUC)

function traverse(InputGroundTruth, InputSaliencyMap, OutputResultsAUC)
idsGroundTruth = dir(InputGroundTruth);
for i = 1:length(idsGroundTruth)
    if idsGroundTruth(i, 1).name(1)=='.'
        continue;
    end
    if idsGroundTruth(i, 1).isdir==1
         if ~isdir(strcat(OutputResultsAUC, idsGroundTruth(i, 1).name, '/'))
            mkdir(strcat(OutputResultsAUC, idsGroundTruth(i, 1).name, '/'));
        end
        traverse(strcat(InputGroundTruth, idsGroundTruth(i, 1).name, '/'), strcat(InputSaliencyMap, idsGroundTruth(i, 1).name, '/'), strcat(OutputResultsAUC, idsGroundTruth(i, 1).name, '/'));
    else
        if strcmp(idsGroundTruth(i, 1).name((end-2):end), 'mat' )
            eval(['load', ' ', strcat(InputGroundTruth, idsGroundTruth(i, 1).name)]);
            subidsSaliencyMap = dir(InputSaliencyMap);
            for curAlgNum = 3:length(subidsSaliencyMap)
                %outFileNameROC = strcat(OutputResultsROC, subidsSaliencyMap(curAlgNum, 1).name, '.mat');
                outFileNameAUC = strcat(OutputResultsAUC, subidsSaliencyMap(curAlgNum, 1).name, '.mat');
                subsubidsSaliencyMap = dir(strcat(InputSaliencyMap, subidsSaliencyMap(curAlgNum, 1).name, '/'));
                %% compute the number of images in the dataset
                imgNum = 0;
                for curImgNum = 3:length(subsubidsSaliencyMap)
                    if strcmp(subsubidsSaliencyMap(curImgNum, 1).name((end-2):end), 'jpg' )||...
                            strcmp(subsubidsSaliencyMap(curImgNum, 1).name((end-2):end), 'png' )||...
                            strcmp(subsubidsSaliencyMap(curImgNum, 1).name((end-2):end), 'bmp' )||...
                            strcmp(subsubidsSaliencyMap(curImgNum, 1).name((end-2):end), 'tif' )||...
                            strcmp(subsubidsSaliencyMap(curImgNum, 1).name((end-3):end), 'jpeg' )
                        imgNum = imgNum+1;
                    end
                end
                %%
%                 tp = cell(1, imgNum);
%                 fp = cell(1, imgNum);
                AUCscore = cell(1, imgNum);
                
                for curImgNum = 3:(imgNum+2)
                    curGroundTruth = double(imgCell{curImgNum-2,1});
                    curSaliencyMap = double(imread(strcat(InputSaliencyMap, subidsSaliencyMap(curAlgNum, 1).name, '/', subsubidsSaliencyMap(curImgNum, 1).name)));
%                     [curtp, curfp, curAUCscore] = AUC_Judd(curSaliencyMap, curGroundTruth);
                    curAUCscore = AUC_Judd(curSaliencyMap, curGroundTruth);
%                     tp{curImgNum-2} = curtp;
%                     fp{curImgNum-2} = curfp;
                    AUCscore{curImgNum-2} = curAUCscore;
                end
%                 tp = mean(cell2mat(tp), 2);
%                 fp = mean(cell2mat(fp), 2);
                AUCscore = mean(cell2mat(AUCscore), 2);
%                 save(outFileNameROC, 'tp', 'fp');
                saveAUCscore = strcat('AUCscore', '_', subidsSaliencyMap(curAlgNum).name);
                eval([saveAUCscore, '=', 'AUCscore']);
                save(outFileNameAUC, saveAUCscore);
            end
        end
        break;
    end
end
%% END CVBIOUC

function score = AUC_Borji(saliencyMap, fixationMap, Nsplits, stepSize, toPlot)
% saliencyMap is the saliency map
% fixationMap is the human fixation map (binary matrix)
% Nsplits is number of random splits
% stepSize is for sweeping through saliency map
% if toPlot=1, displays ROC curve

if nargin < 5, toPlot = 0; end
if nargin < 4, stepSize = .1; end
if nargin < 3, Nsplits = 100; end

% If there are no fixations to predict, return NaN
if ~any(fixationMap)
    score=NaN;
    disp('no fixationMap');
    return
end 

% make the saliencyMap the size of the image of fixationMap
if size(saliencyMap, 1)~=size(fixationMap, 1) || size(saliencyMap, 2)~=size(fixationMap, 2)
    saliencyMap = imresize(saliencyMap, size(fixationMap));
end

% normalize saliency map
saliencyMap = (saliencyMap-min(saliencyMap(:)))/(max(saliencyMap(:))-min(saliencyMap(:)));

S = saliencyMap(:);
F = fixationMap(:);

Sth = S(F>0); % sal map values at fixation locations
Nfixations = length(Sth);
Npixels = length(S);

% for each fixation, sample Nsplits values from anywhere on the sal map
r = randi([1 Npixels],[Nfixations,Nsplits]);
randfix = S(r); % sal map values at random locations

% calculate AUC per random split (set of random locations)
auc = nan(1,Nsplits);
for s = 1:Nsplits
    
    curfix = randfix(:,s);
    
    allthreshes = fliplr([0:stepSize:max([Sth;curfix])]);
    tp = zeros(length(allthreshes)+2,1);
    fp = zeros(length(allthreshes)+2,1);
    tp(1)=0; tp(end) = 1; 
    fp(1)=0; fp(end) = 1; 
    
    for i = 1:length(allthreshes)
        thresh = allthreshes(i);
        tp(i+1) = sum((Sth >= thresh))/Nfixations;
        fp(i+1) = sum((curfix >= thresh))/Nfixations;
    end

    auc(s) = trapz(fp,tp);
end

score = mean(auc); % mean across random splits

if toPlot
    subplot(121); imshow(saliencyMap, []); title('SaliencyMap with fixations to be predicted');
    hold on;
    [y, x] = find(fixationMap);
    plot(x, y, '.r');
    subplot(122); plot(fp, tp, '.b-');   title(['Area under ROC curve: ', num2str(score)])
end

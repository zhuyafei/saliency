%% CVBIOUC
function benchAUC_Judd()
InputGroundTruth = './tmp/Datasets/GroundTruth/';
InputSaliencyMap = './tmp/SaliencyMaps/';
%OutputResultsROC = './tmp/Results/roc/';
OutputResultsAUC = './tmp/Results/aucJudd/';
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

function AUCscore = AUC_Judd(saliencyMap, fixationMap, jitter, toPlot)
% saliencyMap is the saliency map
% fixationMap is the human fixation map (binary matrix)
% jitter = 1 will add tiny non-zero random constant to all map locations
% to ensure ROC can be calculated robustly (to avoid uniform region)
% if toPlot=1, displays ROC curve

if nargin < 4, toPlot = 0; end
if nargin < 3, jitter = 1; end

% If there are no fixations to predict, return NaN
if ~any(fixationMap)
    AUCscore=NaN;
    disp('no fixationMap');
    return
end 

% make the saliencyMap the size of the image of fixationMap
if size(saliencyMap, 1)~=size(fixationMap, 1) || size(saliencyMap, 2)~=size(fixationMap, 2)
    saliencyMap = imresize(saliencyMap, size(fixationMap));
end

% jitter saliency maps that come from saliency models that have a lot of
% zero values.  If the saliency map is made with a Gaussian then it does 
% not need to be jittered as the values are varied and there is not a large 
% patch of the same value. In fact jittering breaks the ordering 
% in the small values!
if jitter
    % jitter the saliency map slightly to distrupt ties of the same numbers
    saliencyMap = saliencyMap+rand(size(saliencyMap))/10000000;
end

% normalize saliency map
saliencyMap = (saliencyMap-min(saliencyMap(:)))/(max(saliencyMap(:))-min(saliencyMap(:)));

S = saliencyMap(:);
F = fixationMap(:);
   
Sth = S(F>0); % sal map values at fixation locations
Nfixations = length(Sth);
Npixels = length(S);

allthreshes = sort(Sth, 'descend'); % sort sal map values, to sweep through values
tp = zeros(Nfixations+2,1);
fp = zeros(Nfixations+2,1);
tp(1)=0; tp(end) = 1; 
fp(1)=0; fp(end) = 1;

for i = 1:Nfixations
    thresh = allthreshes(i);
    aboveth = sum(S >= thresh); % total number of sal map values above threshold
    tp(i+1) = i / Nfixations; % ratio sal map values at fixation locations above threshold
    fp(i+1) = (aboveth-i) / (Npixels - Nfixations); % ratio other sal map values above threshold
end 

AUCscore = trapz(fp,tp);

if toPlot
    subplot(121); imshow(saliencyMap, []); title('SaliencyMap with fixations to be predicted');
    hold on;
    [y, x] = find(fixationMap);
    plot(x, y, '.r');
    subplot(122); plot(fp, tp, '.b-');   title(['Area under ROC curve: ', num2str(score)])
end

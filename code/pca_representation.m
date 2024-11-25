clear all; close all; fclose all;
tic;

c=112;  % column size (number of rows)
r=92;   % row size (number of columns)
ImSize = c*r;
NoImages = 40;
NoFeatures = 30;

fprintf('\n***************** PCA via covariance matrix *****************\n\n');
fidList = fopen('./images/trainlist','r');
  Y = []; % the Data Matrix
  for i = 1:NoImages
	fname = fgetl(fidList);
    image = imread(['./images/' fname]);
    image = reshape(image, ImSize, 1);
	Y = [Y double(image)];
  end
fclose(fidList);

% Mean Vector
MeanFace = (sum(Y')/NoImages)';
%MeanFace = mean(Y, 2);
figure, imshow(reshape(MeanFace, c, r), []);

% Center the Data Matrix
Y = Y - MeanFace*ones(size(Y(1,:)));

% find out the rank of the Data Matrix
fprintf('\nThe rank of the Data Matrix is: %d \n', rank(Y)); 

%step 1: eig. mxm;
[evect, eval] = eig(Y' * Y / (NoImages-1));

%step 2: eig. nxn;
Evect = Y*evect;  % eigenvectors or eigenfaces

%step 3: normalization
[Eval, idx] = sort(diag(eval), 'descend');
for i=1:NoImages
	Evect_norm(:, i) = Evect(:, idx(i))/norm(Evect(:, idx(i)));
end

% display the EigenVectors:
for i=1:NoImages-1
    subplot(5, 8, i);
	imshow(reshape(Evect_norm(:, i), c, r), []);
	title(sprintf('eVector: %d', i));
end

% display the EigenValues:
figure, bar(Eval);
title('EigenValues');

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n***************** PCA for Image Compression *****************\n');
input('hit a key to continue...');
close all; 
% Example: image compression

while(1)

NoFeatures = input('\ninput number of features to store per image: ');
if NoFeatures == 0 break; end
figure;
for i = 1:NoImages
   ginput(1);
   image1 = Y(:, i);
   subplot(1,2,1); imshow(reshape(image1+MeanFace, c, r), []);
   title(sprintf('Original Image: %d', i));
   numbers2store = Evect_norm(:, 1:NoFeatures)'*image1;

   % recover the image:
   image1_recover = Evect_norm(:, 1:NoFeatures)*numbers2store + MeanFace;
   subplot(1,2,2); imshow(reshape(image1_recover, c, r), []);
   title(sprintf('Recovered Image: %d', i));
end

fprintf('\nCompression Ratio = %f \n', ImSize/NoFeatures/4);

end
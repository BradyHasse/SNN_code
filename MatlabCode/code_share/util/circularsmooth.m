function causal_smooth = circularsmooth(causal_in2,gaussize)

SCI = size(causal_in2);
sqsz = max(SCI(1:2));
smsz = min(SCI(1:2));

PD_N = linspace(0, 2*pi*3-((2*pi)/sqsz), sqsz*3);
TargetDir_N = linspace(0, 2*pi*3-((2*pi)/smsz), smsz*3);

causal_in3 = repmat(causal_in2, 3,3,1,1,1);
causal_smooth = zeros([sqsz,sqsz,SCI(3:5)]);%targets x input x window x inputgroup x epoch
for i = 1:SCI(3)%window
    for j = 1:SCI(4)%inputgroup
        for k = 1:SCI(5)%epoch
            mat_ = causal_in3(:,:,i,j,k);
            if j== 4%non-directional input just repeats end values instead of all values
                mat_(:,1:sqsz,:) = repmat(mat_(:,sqsz+1,:),1,sqsz,1);
                mat_(:,(sqsz*2+1):end,:) = repmat(mat_(:,sqsz*2,:),1,sqsz,1);
            end
            mat_ = interp1(TargetDir_N, mat_,PD_N,'makima','extrap');
            mat_ = smoothdata(smoothdata(mat_, 1, "gaussian",gaussize), 2, "gaussian",gaussize);
            mat_ = mat_((sqsz+1):(sqsz*2),(sqsz+1):(sqsz*2),:);
            causal_smooth(:,:,i,j,k) = mat_;
        end
    end
end
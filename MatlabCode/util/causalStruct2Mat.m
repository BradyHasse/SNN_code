function [Mat, sMat] = causalStruct2Mat(causal_in)
    fields = fieldnames(causal_in);
    causal_in2 = [];
    for g = 1:length(fields)
        causal_in2 = cat(5,causal_in2,causal_in.(fields{g}));
    end
    Mat = permute(causal_in2,[2,3,1,4,5]);
    sMat = size(Mat);
end








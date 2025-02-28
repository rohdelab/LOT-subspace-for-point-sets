function [PL_Wgt,WC_T_P,var1] = LOT_LinearEmb(tem_P,tem_Pl,P,Pl,paral)

% This function implements the LOT embedding computation. Given a data sets
% of N images (after particle approximation) as well as a template image.
% This function will compute the LOT embedding for this data set wrt this
% template.
% Input:
% Pl and P: The particle location and weights from the data sets
% tem_P and tem_Pl: The particle location and weights for the template
% image, with M particles
% Output:
% PL_Wgt: The weight for the linear embedding for data set (1*M)
% WC_T_P: The location of the linear embedding (2*M)
% var1: The variance wrt each linear embedding position (usually not used)
% Author: Wei Wang

alpha = 10000;
thd = 5e-3;
dim_T = length(tem_P);

if iscell(P)
    numObj = size(Pl,2);
    if paral
        parfor i = 1:numObj-1
%             display(['Solving linear programming for image #',num2str(i)])
%             
%             [dist_T_P,map_T_P] = ot_scaling_linear(tem_P,tem_Pl,P{i},Pl{i},alpha);
%             f_T_P = reshape(map_T_P,[length(map_T_P)/dim_T,dim_T]);
%             loca = Pl{i};
%             C_T_P = Pl{i}*f_T_P;
%             
%             Cn_T_P=C_T_P./repmat(sum(f_T_P),[size(C_T_P,1),1]);
%             
%             Cw_T_P(:,i) = Cn_T_P(:);
%             WC_T_P{i} = Cn_T_P;
%             
%             var1=[];
            
            display(['Solving linear programming for image #',num2str(i)])
            
            [dist_T_P,map_T_P] = ot_scaling_linear(tem_P,tem_Pl,P{i},Pl{i},alpha);
            f_T_P = reshape(map_T_P,[length(map_T_P)/dim_T,dim_T]);
            loca = Pl{i};
            C_T_P = Pl{i}*f_T_P;
            
            Cn_T_P=C_T_P./repmat(sum(f_T_P),[size(C_T_P,1),1]);
            
            Cw_T_P(:,i) = Cn_T_P(:);
            WC_T_P{i} = Cn_T_P;
            
            var1=[];
        end
        display(['Solving linear programming for image #',num2str(numObj)])
        [dist_T_P,map_T_P] = ot_scaling_linear(tem_P,tem_Pl,P{numObj},Pl{numObj},alpha);
        f_T_P = reshape(map_T_P,[length(map_T_P)/dim_T,dim_T]);
        loca = Pl{numObj};
        C_T_P = Pl{numObj}*f_T_P;
        
        Cn_T_P=C_T_P./repmat(sum(f_T_P),[size(C_T_P,1),1]);
        
        Cw_T_P(:,numObj) = Cn_T_P(:);
        WC_T_P{numObj} = Cn_T_P;
        
        var1=[];
    else
        for i = 1:numObj
            display(['Solving linear programming for image #',num2str(i)])
            
            [dist_T_P,map_T_P] = ot_scaling_linear(tem_P,tem_Pl,P{i},Pl{i},alpha);
            f_T_P = reshape(map_T_P,[length(map_T_P)/dim_T,dim_T]);
            loca = Pl{i};
            C_T_P = Pl{i}*f_T_P;
            
            for j = 1:size(C_T_P,2)
                Cn_T_P(:,j) = C_T_P(:,j)/sum(f_T_P(:,j));
            end
            
            Cw_T_P(:,i) = Cn_T_P(:);
            WC_T_P{i} = Cn_T_P;
            for va = 1:size(f_T_P,2)
                ind = vec(find((f_T_P(:,va)/sum(f_T_P(:,va)))>thd));
                var1(va,i) = std(L2_distance(loca(:,ind),Cn_T_P(:,va)));%loca(:,ind)
            end
            clear Cn_T_P;
        end
    end
else
    [dist_T_P,map_T_P] = ot_scaling_linear(tem_P,tem_Pl,P,Pl,alpha);
    f_T_P = reshape(map_T_P,[length(map_T_P)/dim_T,dim_T]);
    loca = Pl;
    C_T_P = Pl*f_T_P;
    
    for j = 1:size(C_T_P,2)
        Cn_T_P(:,j) = C_T_P(:,j)/sum(f_T_P(:,j));
    end
    Cw_T_P = Cn_T_P(:);
    WC_T_P = Cn_T_P;
    for va = 1:size(f_T_P,2)
        ind = vec(find((f_T_P(:,va)/sum(f_T_P(:,va)))>thd));
        var1(va) = std(L2_distance(loca(:,ind),Cn_T_P(:,va)));%loca(:,ind)
    end
    clear Cn_T_P;
    
end

for j = 1:size(C_T_P,2)
    PL_Wgt(j) = sum(f_T_P(:,j));
end

%   distance(i,j) = ((P_mn)')*(diag(L2_distance(WC_T_P{i},WC_T_P{j})).^2);
clc
clear all
close all

% PROPOSED METHOD
strtH=tic;

%%
tag=0; innm='SYNTH';
cls=[0:4]; x_ax=[1,2,3,4,5];
Nms=100; Ndim=2;

%%
p0=pwd; cd ..; p1=pwd; cd(p0);

inp=[p1 '/DATA/' innm '/'];
outp=[p1 '/RESULTS/' innm '/'];
mdp=[p1 '/DATA/METADATA/' innm '/'];

%%
if exist([inp 'training/metaLOTP/'])
else
    mkdir([inp 'training/metaLOTP/']);
end
if exist([inp 'testing/metaLOTP/'])
else
    mkdir([inp 'testing/metaLOTP/']);
end
if exist(outp)
else
    mkdir(outp);
end

%%
paral=1; Nrep=10; Ntr=length(x_ax); Ncl=length(cls);
force_tr_Lotp=0; force_te_Lotp=0;

Uvec=1; % Deformation modeling; 1 - do enrichment, 0 - no enrichment

%%
accs=[]; cnt_display=0;
for bbb=1:Ntr
    for ccc=1:Nrep

        %% TRAINING
        basis=[]; Pl_tem_all=[]; len_sub=0; %inf; %0; % inf;
        for a=1:length(cls)
            cl=cls(a);

            load([inp 'training/dataORG_' num2str(cl) '.mat']);
            load([inp 'training/dataORG_' num2str(cl) '_tem.mat']);
            xO=xxO; lO=label;

            indr = hdf5read([mdp 'indices_ntr_' num2str(x_ax(bbb)) '_ncl_' num2str(Ncl) '_rep_' num2str(ccc-1) '.hdf5'],'train_index');
            ind=indr(:,a)+1;

            xxO=xO(:,:,ind); label=lO(ind);

            Pl=[]; P=[];
            for b=1:size(xxO,3)
                Pl{b}=xxO(:,:,b); P{b}=ones(Nms,1);
            end
            Pl_tem=xxO_tem; P_tem=ones(Nms,1);
            Pl_tem_all{a}=Pl_tem;


            if exist([inp 'training/metaLOTP/dataLOTP_' num2str(cl) '_ntr_' num2str(x_ax(bbb)) '_ncl_' num2str(Ncl) '_rep_' num2str(ccc-1) '.mat']) & force_tr_Lotp==0
                load([inp 'training/metaLOTP/dataLOTP_' num2str(cl) '_ntr_' num2str(x_ax(bbb)) '_ncl_' num2str(Ncl) '_rep_' num2str(ccc-1) '.mat']);
            else
                [~,LOT_coord,~]=LOT_LinearEmb(P_tem,Pl_tem,P,Pl,paral);
                save([inp 'training/metaLOTP/dataLOTP_' num2str(cl) '_ntr_' num2str(x_ax(bbb)) '_ncl_' num2str(Ncl) '_rep_' num2str(ccc-1) '.mat'],'LOT_coord','-v7.3');
            end

            v=[];
            for b=1:size(LOT_coord,2)
                v{b}=LOT_coord{b}-Pl_tem;
            end



            if Uvec==1
                dtVec=[];
                for b=1:size(LOT_coord,2)
                    tmp_v=v{b}; tmp_v=tmp_v-mean(tmp_v,2); % %
                    tmp_d=LOT_coord{b}; tmp_d=tmp_d-mean(tmp_d,2); % %

                    tmp=tmp_d; % -Pl_tem;

                    UD0=tmp(:);

                    UD=[];
                    for c=1:Ndim
                        UDx=zeros(Ndim,Nms);
                        UDx(c,:)=tmp_d(c,:);
                        UDx=UDx(:);
                        UD=[UD UDx];
                    end

                    US=[];
                    for c=1:Ndim
                        for d=1:Ndim
                            USx=zeros(Ndim,Nms);
                            if c==d
                                continue;
                            else
                                USx(c,:)=tmp_d(d,:);
                            end
                            USx=USx(:);
                            US=[US USx];
                        end
                    end
                    dtVec=[dtVec UD0 UD US];
                end

                UT=[];
                for b=1:Ndim
                    UTx=zeros(Ndim,Nms);
                    UTx(b,:)=1;
                    UTx=UTx(:);
                    UT=[UT UTx];
                end
                dtVec=[dtVec UT];

            else

                dtVec=[];
                for b=1:size(LOT_coord,2)
                    tmp_v=v{b}; tmp_v=tmp_v(:);
                    dtVec(:,b)=tmp_v;
                end
            end


            dtVec(:,isnan(sum(dtVec)))=[]; % % %


            % %
            for aa=1:size(dtVec,2)
                dtVec(:,aa)=dtVec(:,aa)/norm(dtVec(:,aa));
            end
            % %


            [uu,su,vu]=svd(dtVec);
            indx=1:size(LOT_coord,2); s=diag(su); s=s*100/sum(s); eps=1; indx=find(s>eps);
            V=uu;
            basis(cl+1).V = V;

            if len_sub<length(indx)  % len_sub>length(indx); len_sub<length(indx)
                len_sub=length(indx);
            end

            disp(' '); disp([innm ' - Training completed: ' num2str(a*100/length(cls)) ' %']); disp(' ');
        end

        %% TESTING
        xte=[]; yte=[];
        for a=1:length(cls)
            cl=cls(a);
            load([inp 'testing/dataORG_' num2str(cl) '.mat']);
            xte=cat(3,xte,xxO); yte=[yte;label(:)];
        end
        Pl=[]; P=[];
        for a=1:size(xte,3)
            Pl{a}=xte(:,:,a);
            P{a}=ones(Nms,1);
        end

        D=[];
        for a=1:length(cls)
            cl=cls(a);
            Pl_tem=Pl_tem_all{a};

            if exist([inp 'testing/metaLOTP/dataLOTP_xx_tr_' num2str(cl) '.mat']) & force_te_Lotp==0
                load([inp 'testing/metaLOTP/dataLOTP_xx_tr_' num2str(cl) '.mat'])
            else
                [~,LOT_coord,~]=LOT_LinearEmb(P_tem,Pl_tem,P,Pl,paral);
                save([inp 'testing/metaLOTP/dataLOTP_xx_tr_' num2str(cl) '.mat'],'LOT_coord','-v7.3');
            end

            B = basis(cl+1).V; B=B(:,1:len_sub);

            Xtest=[];
            for b=1:size(LOT_coord,2)
                tmp_v=LOT_coord{b}-Pl_tem; tmp_v=tmp_v(:);
                tmp_d=LOT_coord{b}; tmp_d=tmp_d(:);
                Xtest(:,b)=tmp_d; % tmp_v;
            end
            Xproj = (B*B')*Xtest;
            Dproj = Xtest - Xproj;
            D(a,:) = sqrt(sum(Dproj.^2,1));

            disp(' '); disp([innm ' - Test completed: ' num2str(a*100/length(cls)) ' %']); disp(' ');
        end

        %%
        cnt_display=cnt_display+1;
        disp(['Progress: ' num2str(100*cnt_display/(Nrep*Ntr)) ' % (' num2str(cnt_display) ' of ' num2str(Nrep*Ntr) ' ) COMPLETED.....................................'])

        %%
        [~,ypred] = min(D); ypred=cls(ypred); ypred=ypred(:);
        accs_t=numel(find(ypred==yte))/length(ypred);
        Accuracy = accs_t*100;
        disp(['Test accuracy: ' num2str(Accuracy) ' %'])
        accs(ccc,bbb)=accs_t;

    end
end

mean(accs)

save([outp 'proposed.mat'],'accs');
endH=toc(strtH)




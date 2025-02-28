function [distance,mapping,t_con,c] = ot_scaling_linear(P_o,Pl_o,Q_o,Ql_o,alpha)

% Computing the optimal transportation distance between two set of
% delta masses specified in input arguments. 

% INPUTS:
% P_o: weights of first set of particles
% Pl_o: coordinates of first set of particles [[x1,y1];[x2,y2];...]
% Q_o: weights of second set of particles
% Ql_o: coordinates of second set of particles
% alpha: constant specifying by how much to weigh the intensity
% differences between P_o and Q_o (set to zero if to ignore
% intensity differences.

% Author G.K. Rohde
% Last modified: Wei Wang
% basic setups
P = vec(P_o./sum(P_o));
Q = vec(Q_o./sum(Q_o));
Pl = Pl_o;
Ql = Ql_o;

Np = length(P);
Nq = length(Q);

% setup constraints (this must be sped up, wasteful in how it handles
% memory)
c = zeros(Np*Nq,1);
x = c;
b = [P;Q];
lbound = x*0;
A = sparse([]);
kp = ones(1,Nq);
Arow = zeros(1,Nq*Np);
k_index = 1:Nq:Np*(Nq);

% % Computing the distance between point masses using LOOP (Slow)
% counter = 1;
% for i=1:Np
%     for j=1:Nq
%         c(counter) = norm(Pl(:,i) - Ql(:,j))^2; %distance
%         counter = counter+1;
%     end
% end

% Computing the distance between point masses using Matrix multiplication(Fast)
c_2 = L2_distance(Ql,Pl).^2;
c = c_2(:);

A = sparse(Np+Nq,Np*Nq);
for i=1:Np
    c_row = Arow;
    c_row( (i-1)*Nq+1 : (i-1)*Nq+Nq ) =  kp;
    A(i,:) = sparse(c_row);
end

for i= (Np+1):(Np+Nq)
   c_row = Arow;
   w = k_index+(i-Np-1);
   c_row(w) = 1;
   A(i,:) = sparse(c_row);
end
tic
% options = optimset('LargeScale','on','Display','off','MaxIter',2000,'TolFun',0.000001);
options = optimset('LargeScale','on','Display','off','MaxIter',1000,'TolFun',1e-6,'algorithm','interior-point');
mapping = linprog(c,[],[],A,b,lbound,[],options); % % %
t_con = toc;
display(['OT plan calculated in t=',num2str(t_con),' sec'])

% % % 

maxmc=max(length(mapping),length(c));
t1=zeros(maxmc,1); t2=t1;
t1(1:length(mapping))=mapping; mapping=t1;
t2(1:length(c))=c; c=t2;
d_map = (mapping'*c)^2;
% % % d_map = (mapping'*c)^2;

% % %

d_int = alpha*((log(sum(P_o)/sum(Q_o))))^2;
distance = sqrt(d_map + d_int);

clear all; close all;


aa = 40;        % Side length of square grid
N = aa*aa;      % Number of neurons
tlen = 20000;
%tdiv = 50;
msize = tlen;%/tdiv +1;

x = zeros(1,N);
xs = zeros(msize,N);
z = x;
We = (eye(N));
Wi = We;


h = 0.001;
B = 10; C = 10;


Ae = 0.6;      % Excitatory kernal max height
ke = 1;        % Excitatory kernel width
Ai = 0.02;   % Inhibitory kernal max height (Play with this parameter!)
ki = 3;        % Inhibitory kernal width


ra = 0.01;
recov = 1.2; % Play with this parameter :)
rstrength = 0.5; % And this one
zinh = 10;
toff = 0.4;
params = [aa h B C Ae ke Ai ki ra recov rstrength zinh tlen toff];


distan = zeros(N);
kk = 0;
pos = zeros(N,2);
for ii = 1:aa
    for jj = 1:aa
        kk = kk +1;
        pos(kk,:) = [ii jj];
    end
end
tic
for ii = 1:N
    for jj = ii:N
        distan(ii,jj) = norm(squeeze(pos(ii,:)) - squeeze(pos(jj,:)),2);
        distan(jj,ii) = distan(ii,jj);
        Wi(ii,jj) = Ai.*exp(-(distan(ii,jj)./ki).^2);%./(ki*sqrt(2*pi));
        We(ii,jj) = Ae.*exp(-(distan(ii,jj)./ke).^2);% - Ae.*exp(-(distan(ii,jj)./(0.85*ke)).^2);%./(ki*sqrt(2*pi));
        We(jj,ii) = We(ii,jj);
        Wi(jj,ii) = Wi(ii,jj);
        Wi(ii,ii) = 0;
        %We(ii,ii) = 0;
        
    end
end

toc

%Wi = 0.*Wi;
%Inpshape = (1:aa)'*(1:aa)/N;
%Inpshape = ((1:aa)./aa)'*ones(1,aa);
%Inpshape  = zeros(aa);
%Inpshape(10,10) = 2.75;
%Inpshape(11,14) = 2.75;

%Inpshape((aa/2-2):(aa/2+2),(aa/2-2):(aa/2+2)) = 0.03;

%Inpshape(1:3,1:3) = 0.03;
%Inpshape = 1.5*round(max(rand(aa)-0.492,0));
%if max(max(Inpshape))<0.1
%    Inpshape(2,2) =1;
%end

Inp = 0.1*round(max(rand(tlen,N) - 0.499,0));

% t1 = 1+round(rand*(tlen*.25));
% t2 = 1+round(rand*(tlen*.25));
% Inp(t1:t1+1000,(1+round(rand*(N-1)))) = 0.1;
% Inp(t2:t2+1000,(1+round(rand*(N-1)))) = 0.1;
%Inp = 0.1*(rand(tlen,N));
tic

Inp((tlen*toff):tlen,:)=0;
for t = 1:tlen
    %ii = t;
    
    %Inp(t,:) = fac(t).*Inpshape(:);
    %td = max(1, t-500);
    y = max(x,0);
    %inh = xs(td,:);
    inh = y;
    
    x = x + h.*((B-x).*(Inp(t,:)  + (y)*We) - (x+C).*( (y*Wi) + rstrength.*z) -x*ra );
    
    x = max(x,0);
    x = min(B,x);
    
    %z= z + rr.*h.*(x -z);
    z = z + recov*h*((B-z).*inh - zinh*z.*(1-sign(inh)));
    z = max(z,0);
    z= min(B,z);
    
    
    xs(t,:) = max(x,0);
end
toc

%scrsz = get(0,'ScreenSize');
%figure('Position',[10 scrsz(4)/2-400 scrsz(3)/2 scrsz(4)/2+300])
figure
subplot(3,1,1)
imagesc(Inp')
title('Input')

subplot(3,1,2)
imagesc(xs');
title('Activities')

subplot(3,1,3)
plot(xs);
title('Activities')

% subplot(4,1,4)
% plot(1:N,Wi(:,round(N/2)),'r.',1:N,We(:,round(N/2)));
% title('Connection weights')

figure
inde = round(N/2 - aa/2);
subplot(1,2,1)
imagesc(reshape(We(inde,:),aa,aa))
axis equal
axis off
title('Excitatory kernel')
subplot(1,2,2)
imagesc(reshape(Wi(inde,:),aa,aa))
axis equal
axis off
title('Inhibitory kernel')

%gifmaker_2D(xs,aa,file_name,params)
show2D(xs,Inp,aa)

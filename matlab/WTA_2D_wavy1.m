clear all; close all;

aa = 20;
N = aa*aa;
tlen = 20000;

x = zeros(1,N);
xs = zeros(tlen,N);

%S = sin(0.05.*(1:tlen))+1;

%Inp = (S'*ones(1,N)).*rand(tlen,N);
We = zeros(N);
Wi = We;

 Ae = 0.7;
 ke = 1;
 Ai = 0.0085; ki = 7;
 
Inp = zeros(tlen,N);
 
distan = zeros(N);
kk = 0;
pos = zeros(N,2);
for ii = 1:aa
    for jj = 1:aa
        kk = kk +1;
      pos(kk,:) = [ii jj];
    end
end

for ii = 1:N
    for jj = ii:N
        distan(ii,jj) = norm(squeeze(pos(ii,:)) - squeeze(pos(jj,:)),2);
        %distan(ii,jj) = abs(ii-jj);
        distan(jj,ii) = distan(ii,jj);
        Wi(ii,jj) = Ai.*exp(-(distan(ii,jj)./ki).^2);%./(ki*sqrt(2*pi));
        We(ii,jj) = Ae.*exp(-(distan(ii,jj)./ke).^2);% - Ae.*exp(-(distan(ii,jj)./(0.85*ke)).^2);%./(ki*sqrt(2*pi));
        We(jj,ii) = We(ii,jj);
        Wi(jj,ii) = Wi(ii,jj);
        Wi(ii,ii) = 0;
        %We(ii,ii) = 0;
        
    end
end

%Wi = 0.*Wi;

%Inp = 0.01.*rand(tlen,N);
Inp = round(max(rand(tlen,N) - 0.45,0));
h = 0.001;
B = 10; C = 10;
z = x;
p =10;

ra = 0.01;
rr = 2.01;
f = 1;

Inpy = round(max(rand(1,N)-0.49,0));

%Inp(tlen*0.2:tlen,:) = 0;
for t = 1:tlen
   Inp(t,:) = Inpy;
    %td = max(1, t-500);
   y = max(x,0);
   %inh = xs(td,:);
   inh = y;
   
    x = x + h.*((B-x).*(Inp(t,:)  + (y)*We) - (x+C).*( (y*Wi) + 2.*z) -x*ra );
   
    x = max(x,0);
    x = min(B,x);
    
    %z= z + rr.*h.*(x -z);
    z = z + 0.3*h*((B-z).*inh - 10*z.*(1-sign(inh)));
    z = max(z,0);
    z= min(B,z);
    
    xs(t,:) = max(x,0);
end
scrsz = get(0,'ScreenSize');
figure('Position',[10 scrsz(4)/2-400 scrsz(3)/2 scrsz(4)/2+300])
subplot(5,1,1)
%plot(Inp)

imagesc(Inp')
subplot(5,1,2)
%plot(max(xs,0));
imagesc(xs');

subplot(5,1,3)
%plot(1:tlen,sum(Inp'))
plot(Inp)

subplot(5,1,4)
%plot(sum(xs'));
plot(xs);

subplot(5,1,5)
plot(1:N,Wi(:,round(N/2)),'r.',1:N,We(:,round(N/2)));

sho3Da(xs,Inp,aa)

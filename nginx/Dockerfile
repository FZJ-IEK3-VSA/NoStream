FROM nginx:alpine
RUN rm /etc/nginx/conf.d/default.conf

COPY /certificates                      /home/debian/certificates
COPY /NoStream/nginx/nginx.prod.conf    /etc/nginx/conf.d/default.conf
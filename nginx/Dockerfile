FROM nginx:alpine
RUN rm /etc/nginx/conf.d/default.conf

COPY /certificates                      /etc/nginx/conf.d/certificates
COPY /NoStream/nginx/nginx.prod.conf    /etc/nginx/conf.d/default.conf
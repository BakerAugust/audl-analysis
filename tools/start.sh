docker run -it \
    --expose 8888 \
    --publish=127.0.0.1:8888:8888/tcp \
    -v ${PWD}/src:/src \
    audlanalysis:latest bash
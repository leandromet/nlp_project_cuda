
dd if=/dev/zero of=/mnt/exfat/docker_volume.img bs=1M count=2000000


mkfs.ext4 /data14/bd_dataspace/docker_volume.img

sudo mkdir -p /mnt/docker_volume

sudo losetup -fP /data14/bd_dataspace/docker_volume.img

losetup -a

sudo mount /dev/loop0 /mnt/docker_volume

sudo chown -R 1000:1000 /mnt/docker_volume



docker run   --hostname=bc745628fbc6   --mac-address=02:42:ac:11:00:03   --env=POSTGRES_PASSWORD=password_change   --env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/postgresql/17/bin   --env=GOSU_VERSION=1.17   --env=LANG=en_US.utf8   --env=PG_MAJOR=17   --env=PG_VERSION=17.2-1.pgdg110+1   --env=PGDATA=/var/lib/postgresql/data   --env=POSTGIS_MAJOR=3   --env=POSTGIS_VERSION=3.5.0+dfsg-1.pgdg110+1   --volume=/mnt/docker_volume/pgdatamapbiomas:/var/lib/postgresql/data   --network=bridge   -p 5434:5432   --restart=always   --label='maintainer=PostGIS Project - https://postgis.net'   --label='org.opencontainers.image.description=PostGIS 3.5.0+dfsg-1.pgdg110+1 spatial database extension with PostgreSQL 17 bullseye'   --label='org.opencontainers.image.source=https://github.com/postgis/docker-postgis'   --runtime=runc   -d postgis/postgis:latest


docker ps

ls -l /mnt/docker_volume

psql -h localhost -p 5434 -U postgres

/data14/bd_dataspace/docker_volume.img /mnt/docker_volume ext4 loop 0 0
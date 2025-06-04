#!/usr/bin/env python3
"""
S3 bucket access functions.
"""

import os
import boto3
from config import cfg


class S3Manager:
    def __init__(self):

        self.region_name = cfg.s3.region
        self.aws_access_key_id = cfg.s3.access_key_id
        self.aws_secret_access_key = cfg.s3.secret_access_key
        self.bucket_name = cfg.s3.bucket_name

        try:
            # s3 credentials
            self.s3 = boto3.resource(
                service_name='s3',
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            self.bucket = self.s3.Bucket(self.bucket_name)

            self.s3_client = boto3.client('s3', region_name=self.region_name,
                                          aws_access_key_id=self.aws_access_key_id,
                                          aws_secret_access_key=self.aws_secret_access_key)
        except Exception as e:
            raise Exception(f"S3_CONNECTION_ERROR: {e}")

    def upload_file(self, destination_path, source_path, extra_args=None):
        if extra_args is None:
            extra_args = {}
        extra_args['ContentType'] = 'video/mp4'
        self.s3_client.upload_file(
            source_path,
            self.bucket_name,
            destination_path,
            ExtraArgs=extra_args
        )


    def upload_folder(self, destination_folder, source_folder_path):
        for img in os.listdir(source_folder_path):
            self.s3_client.upload_file(
                                          os.path.join(source_folder_path, img),
                                          self.bucket_name,
                                          os.path.join(destination_folder, img)
                                      )
            
    def delete_object(self, object_key):
        """
        Delete an object from the S3 bucket.
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            print(f"Object {object_key} deleted successfully.")
        except Exception as e:
            raise Exception(f"S3_DELETE_OBJECT_ERROR: {e}")

    def download(self, save_path, s3_file_path):
        self.bucket.download_file(s3_file_path, save_path)

    # Use for get authenticated s3 URL
    def s3_getAuthenticatedUrl(self, content_path, expiration=cfg.s3.ExpiresIn):
        s3Url = self.s3_client.generate_presigned_url('get_object', Params={'Bucket': self.bucket_name, 'Key': content_path}, ExpiresIn=expiration)
        return s3Url

    def get_public_url(self, object_key):
        # Construct and return the public URL for the object
        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_key}"
    

if __name__ == '__main__':
    name_region = cfg.s3.region
    access_key_id = cfg.s3.access_key_id
    secret_access_key = cfg.s3.secret_access_key
    name_bucket = cfg.s3.bucket_name
    downloading_file = 'ZY5GV.mp4'
    save_location = 'ZY5GV.mp4'

    s3_bucket = S3Manager()
    # # s3_bucket.download(save_path=save_location, s3_file_path=downloading_file)
    extra_args = {'ContentType': 'video/mp4'}
    s3_bucket.upload_file(destination_path=downloading_file, source_path=save_location, extra_args=extra_args)
    # url = s3_bucket.s3_getAuthenticatedUrl(content_path=downloading_file, expiration=604800)
    url = s3_bucket.get_public_url(downloading_file)
    print(url)


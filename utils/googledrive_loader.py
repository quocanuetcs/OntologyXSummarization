import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from utils.logger import get_logger

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = os.path.dirname(os.path.realpath(__file__)) \
                                                    + '/../data/googledrive/client_secrets.json'
logger = get_logger(__file__)


class GoogleDriveLoader:
    CREDENTIALS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/googledrive/credentials.json'

    def __init__(self):
        self.gauth = GoogleAuth()
        self.gauth.LoadCredentialsFile(self.CREDENTIALS_PATH)
        if self.gauth.credentials is None:
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            self.gauth.Refresh()
        else:
            self.gauth.Authorize()
        # self.gauth.SaveCredentialsFile(self.CREDENTIALS_PATH)
        self.drive = GoogleDrive(self.gauth)

    def list_file(self, folder_id):
        file_list = self.drive.ListFile({'q': "'" + folder_id + "' in parents and trashed=false"}).GetList()
        return file_list

    def upload_file(self, folder_id, title, content):
        # Delete already exist file
        files = self.list_file(folder_id)
        for file in files:
            if file['title'] == title:
                file.Delete()
                break
        # Upload file
        file = self.drive.CreateFile({
            'title': title,
            'parents': [{'id': folder_id}]
        })
        file.SetContentString(content)
        file.Upload()
        logger.info('Upload file {} successfully'.format(title))

    def get_content_by_title(self, folder_id, title):
        files = self.list_file(folder_id)
        for file in files:
            if file['title'] == title:
                return file.GetContentString()
        return None

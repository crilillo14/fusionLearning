# lowkey don't need this anymore

import os
import cityscapesscripts.download.downloader as downloader

session = downloader.login()

downloader.list_available_packages(session=session)

downloader.download_packages(session=session, package_names=["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"], destination_path=os.path.join(os.path.dirname(__file__), "cityscapes"))
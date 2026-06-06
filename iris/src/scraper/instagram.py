import instaloader
from dataclasses import dataclass
import subprocess
import os
from pathlib import Path
import datetime



# create struct to store collected infos on an instagram post
@dataclass
class PostData:
    shortcode: str
    date: datetime
    location: str
    title: str
    caption: str
    tagged_users: list[str]
    media_urls: list[str]

# create struct to store the collected infos on the target instragram profile
@dataclass
class InstagramData:
    private: bool
    username: str
    full_name: str
    bio: str
    bio_hashtags: list[str]
    bio_mentions: list[str]
    pfp_url: str
    followees_count: int
    followers_count: int
    external_url: str
    media_count: int
    followees: list[str]
    followed_hashtags: list[str]
    posts: list[PostData]


def ft_instaloader(url):

    # extract the target username from the url
    target = url.split("instagram.com/")[1].split("/")[0]
    L = instaloader.Instaloader()

    # ask user if he want to login to use this script
    user_response = ""
    while user_response not in ["y", "n"]:
        user_response = input("Do you want to use instaloader? [y/N]: ").strip().lower()
        if user_response == "":
            user_response = "n"

    # if yes then prompt user for username and try to load session if session file exist but error then delete it
    if user_response == "y":
        print("Please do not use your personnal account")
        try:
            usr = input("Enter your Instagram username: ").strip()
            L.load_session_from_file(usr)
        except FileNotFoundError:
            # if session not created yet then ask user to login on firefox
            print("Please login to instagram on firefox then close the browser")
            is_done = ""
            while is_done != "y":
                is_done = input("Is it done? [y/N]: ").strip().lower()

            # load cookies from firefox
            try:
                subprocess.run(["instaloader", "--load-cookies", "firefox"])
            except Exception as e:
                print(f"Failed to import cookies: {e}")
                return
            
            # load session file if error then delete session file 
            try:
                L.load_session_from_file(usr)
            except Exception as e:
                print(f"Failed to load session: {e}")
                session_file = Path.home().joinpath(".config", "instaloader", f"session-{usr}")
                os.remove(session_file)
                print("Session file deleted, you need to run the script again to create a new one")
                return

        except Exception as e:
            print(f"Failed to load session: {e}")
            session_file = Path.home().joinpath(".config", "instaloader", f"session-{usr}")
            os.remove(session_file)
            print("Session file deleted, you need to run the script again to create a new one")
            return
    else:
        return

    # load target profile
    try:
        profile = instaloader.Profile.from_username(L.context, target)
    except Exception as e:
        print(f"Failed to load target: {e}")
        return

    # collect infos that can be collected even if the account is in private mode
    private = profile.is_private
    username = profile.username
    full_name =  profile.full_name
    bio = profile.biography
    bio_hashtags = profile.biography_hashtags
    bio_mentions = profile.biography_mentions
    pfp_url = profile.profile_pic_url
    followees_count = profile.followees
    followers_count = profile.followers
    external_url = profile.external_url
    media_count = profile.mediacount

    # collect infos that can not be collected if the account is in private mode
    followees = []
    followed_hashtags = []
    posts = []
    if not private:
        # get all followees usernames
        try:
            for followee in profile.get_followees():
                followees.append(followee.username)
        except Exception as e:
            print(f"Followees fetching not completed: {e}")
        
        # get all followed hashtags names
        try:
            for hashtag in profile.get_followed_hashtags():
                followed_hashtags.append(hashtag.name)
        except Exception as e:
            print(f"Followed hashtags fetching not completed: {e}")

        # get all posts
        try:
            for post in profile.get_posts():
                try:
                    shortcode = post.shortcode
                    date = post.date_local
                    try:
                        location = post.location.name if post.location else None
                    except Exception as e:
                        print(f"Failed to fetch post {shortcode} location: {e}")
                        location = None
                    title = post.title if post.title else None
                    caption = post.caption
                    tagged_users = post.tagged_users

                    media_urls = []
                    # if post is a sidecar then iterate over the nodes and check if it is a video or not and get url
                    if post.typename == "GraphSidecar":
                        for node in post.get_sidecar_nodes():
                            if node.is_video:
                                media_urls.append(node.video_url)
                            else:
                                media_urls.append(node.display_url)
                    # if post is not a sidecar then check if it is a video or not and get url
                    else:
                        if post.is_video:
                            media_urls.append(post.video_url)
                        else:
                            media_urls.append(post.url)

                    post_data = PostData(shortcode, date, location, title, caption, tagged_users, media_urls)
                    posts.append(post_data)
                except Exception as e:
                    print(f"Failed to fetch post {shortcode}: {e}")
                    continue
        except Exception as e:
            print(f"Posts fetching not completed: {e}")
    
    return InstagramData(private, username, full_name, bio, bio_hashtags, bio_mentions, pfp_url, followees_count, followers_count, external_url, media_count, followees, followed_hashtags, posts)




if __name__ == "__main__":
    url = "https://www.instagram.com/teeqzyk/"
    res = ft_instaloader(url)
    print(res)
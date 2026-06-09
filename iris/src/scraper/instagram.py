import instaloader
from dataclasses import dataclass, asdict
import subprocess
import os
from pathlib import Path
import urllib
import json



'''
INSTRUCTIONS
    if you encounter the error: "400 Bad Request - "fail" status, message "invalid request" when accessing https://www.instagram.com/graphql/query"
    please follow the instructions provided at this link: https://github.com/instaloader/instaloader/issues/2695#issuecomment-4495719808
'''

# create strcut to store collected infos on instagram highlights
@dataclass
class HighlightData:
    title: str
    date: str
    caption: str
    caption_mentions: list[str]

# create strcut to store collected infos on instagram stories
@dataclass
class StoryData:
    date: str
    caption: str
    caption_mentions: list[str]

# create struct to store collected infos on an instagram post
@dataclass
class PostData:
    date: str
    location: str
    title: str
    caption: str
    tagged_users: list[str]

# create struct to store the collected infos on the target instragram profile
@dataclass
class InstagramData:
    private: bool
    username: str
    full_name: str
    bio: str
    bio_hashtags: list[str]
    bio_mentions: list[str]
    followees_count: int
    followers_count: int
    external_url: str
    media_count: int
    followees: list[str]
    followed_hashtags: list[str]
    posts: list[PostData]
    stories: list[StoryData]
    highlights: list[HighlightData]


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
                print("Fallback on default scraping")
                return url, None
            
            # load session file if error then delete session file 
            try:
                L.load_session_from_file(usr)
            except Exception as e:
                print(f"Failed to load {usr} session: {e}")
                session_file = Path.home().joinpath(".config", "instaloader", f"session-{usr}")
                os.remove(session_file)
                print("Session file deleted")
                print("Fallback on default scraping")
                return url, None

        except Exception as e:
            print(f"Failed to load {usr} session: {e}")
            session_file = Path.home().joinpath(".config", "instaloader", f"session-{usr}")
            os.remove(session_file)
            print("Session file deleted")
            print("Fallback on default scraping")
            return url, None
    else:
        print("Fallback on default scraping")
        return url, None

    # load target profile
    try:
        profile = instaloader.Profile.from_username(L.context, target)
    except Exception as e:
        print(f"Failed to load {target} profile: {e}")
        print("You may want to take a look at the instructions in instagram.py")
        print("Fallback on default scraping")
        return url, None

    # collect infos that can be collected even if the account is in private mode
    private = profile.is_private
    username = profile.username
    full_name =  profile.full_name
    bio = profile.biography
    bio_hashtags = profile.biography_hashtags
    bio_mentions = profile.biography_mentions
    followees_count = profile.followees
    followers_count = profile.followers
    external_url = profile.external_url
    media_count = profile.mediacount
    has_public_story = profile.has_public_story

    # fetch target pfp url then download it
    target_dir = f"./{username}_insta_profile"
    os.makedirs(target_dir, exist_ok=True)
    try:
        pfp_url = profile.profile_pic_url
        urllib.request.urlretrieve(pfp_url, os.path.join(target_dir, f"{username}.jpg"))
    except Exception as e:
        print(f"Failed to download {username} pfp: {e}")

    # collect infos that can not be collected if the account is in private mode
    followees = []
    followed_hashtags = []
    posts = []
    stories = []
    highlights = []
    if not private:
        # ask user if he wants to get target followees usernames
        user_response = ""
        while user_response not in ["y", "n"]:
            user_response = input(f"{username} has {followees_count} followees, do you want to get their usernames? [y/N]: ").strip().lower()
            if user_response == "":
                user_response = "n"

        # if yes then get all followees usernames
        if user_response == "y":
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

        # ask user if he wants to get target posts
        user_response = ""
        while user_response not in ["y", "n"]:
            user_response = input(f"{username} has {media_count} posts, do you want to scrap them? [y/N]: ").strip().lower()
            if user_response == "":
                user_response = "n"

        # if yes then get all posts
        if user_response == "y":
            try:
                for post in profile.get_posts():
                    try:
                        date = str(post.date_local)
                        try:
                            location = post.location.name if post.location else None
                        except Exception as e:
                            print(f"Failed to fetch post location: {e}")
                            location = None
                        title = post.title if post.title else None
                        caption = post.caption
                        tagged_users = post.tagged_users

                        # download all posts
                        os.chdir(target_dir)
                        L.download_post(post, target=f"{username}_posts")
                        os.chdir("..")
    
                        post_data = PostData(date, location, title, caption, tagged_users)
                        posts.append(post_data)
                    except Exception as e:
                        print(f"Failed to fetch post: {e}")
                        continue
            except Exception as e:
                print(f"Posts fetching not completed: {e}")

        if has_public_story:
            # ask user if he wants to get target stories
            user_response = ""
            while user_response not in ["y", "n"]:
                user_response = input(f"{username} has stories, do you want to scrap them? [y/N]: ").strip().lower()
                if user_response == "":
                    user_response = "n"

            # if yes then get stories
            if user_response == "y":
                try:
                    for story in L.get_stories(userids=[profile.userid]):
                        try:    
                            for item in story.get_items():
                                date = str(item.date_local)
                                caption = item.caption
                                caption_mentions = item.caption_mentions

                                # download all stories
                                os.chdir(target_dir)
                                L.download_storyitem(item, target=f"{username}_stories")
                                os.chdir("..")

                            story_data = StoryData(date, caption, caption_mentions)
                            stories.append(story_data)
                        except Exception as e:
                            print(f"Failed to fetch story: {e}")
                            continue
                except Exception as e:
                    print(f"Stories fetching not completed: {e}")
        
        # ask user if he wants to get target highlights
        user_response = ""
        while user_response not in ["y", "n"]:
            user_response = input(f"Do you want to scrap {username} highlights? [y/N]: ").strip().lower()
            if user_response == "":
                user_response = "n"

        # if yes then get all highlights
        if user_response == "y":
            try:
                for highlight in L.get_highlights(profile):
                    try:
                        title = highlight.title
                        for item in highlight.get_items():
                            date = str(item.date_local)
                            caption = item.caption
                            caption_mentions = item.caption_mentions

                            # download all highlights
                            os.chdir(target_dir)
                            L.download_storyitem(item, target=f"{username}_highlights")
                            os.chdir("..")

                            highlight_data = HighlightData(title, date, caption, caption_mentions)
                            highlights.append(highlight_data)
                    except Exception as e:
                        print(f"Failed to fetch highlight: {e}")
                        continue
            except Exception as e:
                print(f"Highlights fetching not completed: {e}")

    # delete all useless file in posts dir to reduce disk usage
    posts_dir = f"./{username}_insta_profile/{username}_posts"
    if os.path.exists(posts_dir):
        posts_files = os.listdir(posts_dir)
        for file in posts_files:
            if file.endswith(".json.xz") or file.endswith(".txt"):
                os.remove(os.path.join(posts_dir, file))

    # delete all useless file in stories dir to reduce disk usage
    stories_dir = f"./{username}_insta_profile/{username}_stories"
    if os.path.exists(stories_dir):
        stories_files = os.listdir(stories_dir)
        for file in stories_files:
            if file.endswith(".json.xz"):
                os.remove(os.path.join(stories_dir, file))

    # delete all useless file in highlights dir to reduce disk usage
    highlights_dir = f"./{username}_insta_profile/{username}_highlights"
    if os.path.exists(highlights_dir):
        highlights_files = os.listdir(highlights_dir)
        for file in highlights_files:
            if file.endswith(".json.xz"):
                os.remove(os.path.join(highlights_dir, file))

    # convert dataclass to json and return it
    return url, json.dumps(asdict(InstagramData(private, username, full_name, bio, bio_hashtags, bio_mentions, followees_count, followers_count, external_url, media_count, followees, followed_hashtags, posts, stories, highlights)), ensure_ascii=False)

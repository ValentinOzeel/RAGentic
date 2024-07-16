import taipy.gui.builder as tgb
from taipy.gui import navigate, notify


def on_login(state, id, login_args):
    username, password = login_args["args"][:2]
    
    if username is None or password is None:
        notify(state, 'error', 'Incorrect username and/or password.', duration=5000)
        return navigate(state, page_ids['unsuccessful_auth'])
    
    else:
        notify(state, 'success', 'Successful authentification.')
        navigate(state, page_ids["success_auth"])

    
    


def on_menu(state, action, info):
    page = info["args"][0]
    navigate(state, to=page)
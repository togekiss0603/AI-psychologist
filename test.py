from jinja2 import Environment, PackageLoader
from consult.consult_main import Consult_Main
import os
from modules import chitchat_bot,medical_bot, classifier
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context
from consult_reply.interactive_conditional_samples import Consult_Reply

from modules import chitchat_bot,medical_bot, classifier
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context
user_msg="情绪好低落啊，能给我一个拥抱吗"

consult_type = Consult_Main()
type = consult_type(user_msg)
print(type)

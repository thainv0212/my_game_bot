import sys

from test_ai import TestAI, MyAI
from machete import Machete
from display_info import DisplayInfo
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
for i in range(2000):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters())
    manager = gateway.entry_point

    manager.registerAI('TestAI', MyAI(gateway, train=True))
    manager.registerAI('Machete', Machete(gateway))
    # manager.registerAI('DisplayInfo', DisplayInfo(gateway))
    print('Start game')
    # game = manager.createGame('ZEN', 'ZEN', 'Machete', 'TestAI', 3)
    game = manager.createGame('ZEN', 'ZEN', 'MctsAi', 'TestAI', 1000)
    manager.runGame(game)
    print('After game')
    sys.stdout.flush()

    print('End of games')
    gateway.close_callback_server()
    gateway.close()
    gateway.shutdown()
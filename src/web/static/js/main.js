function startGame() {
    const config = {
        type: Phaser.AUTO,
        backgroundColor: '#070714',
        scene: [MenuScene, GameScene, GameOverScene],
        parent: 'game-container',
        scale: {
            mode: Phaser.Scale.RESIZE,
        },
        render: {
            antialias: true,
            roundPixels: false,
        },
    };
    new Phaser.Game(config);
}

WebFont.load({
    google: { families: ['Orbitron:700'] },
    active: startGame,
    inactive: startGame,
    timeout: 2000,
});

function highScoreKey(mode, agent) {
    return 'highScore_' + (mode === 'human' ? 'human' : agent);
}

function getHighScore(mode, agent) {
    return parseInt(localStorage.getItem(highScoreKey(mode, agent)) || '0', 10);
}

function saveHighScore(mode, agent, score) {
    const key = highScoreKey(mode, agent);
    const current = getHighScore(mode, agent);
    if (score > current) {
        localStorage.setItem(key, String(score));
        return true;
    }
    return false;
}

function getSFXVolume() {
    const v = parseFloat(localStorage.getItem('sfxVolume'));
    return isNaN(v) ? 0.6 : v;
}

function getMusicVolume() {
    const v = parseFloat(localStorage.getItem('musicVolume'));
    return isNaN(v) ? 0.6 : v;
}

function startGame() {
    const config = {
        type: Phaser.AUTO,
        backgroundColor: '#070714',
        scene: [MenuScene, GameScene, GameOverScene, SettingsScene, HelpScene],
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

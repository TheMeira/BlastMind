class GameOverScene extends Phaser.Scene {
    constructor() {
        super({ key: 'GameOverScene' });
    }

    init(data) {
        this.finalScore  = (data && data.score)        || 0;
        this.bestScore   = (data && data.bestScore)    || this.finalScore;
        this.isNewRecord = (data && data.isNewRecord)  || false;
        this.mode        = (data && data.mode)         || 'human';
        this.agent       = (data && data.agent)        || 'random';
    }

    preload() {
        if (!this.cache.audio.exists('click')) {
            this.load.audio('click', '/static/audio/click.mp3');
        }
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');

        const W = this.scale.width;
        const H = this.scale.height;
        const cx = W / 2;

        this.add.text(cx, H * 0.14, 'GAME OVER', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.09) + 'px',
            fontStyle: 'bold',
            color: '#ff1744',
            shadow: { offsetX: 0, offsetY: 0, color: '#ff1744', blur: 40, fill: true },
        }).setOrigin(0.5);

        if (this.isNewRecord) {
            const rec = this.add.text(cx, H * 0.26, '✦ NEW RECORD ✦', {
                fontFamily: 'Orbitron, Arial',
                fontSize: Math.floor(H * 0.028) + 'px',
                fontStyle: 'bold',
                color: '#ffd740',
                shadow: { offsetX: 0, offsetY: 0, color: '#ff8800', blur: 20, fill: true },
            }).setOrigin(0.5).setAlpha(0);
            this.tweens.add({ targets: rec, alpha: 1, duration: 400, ease: 'Power2' });
            this.tweens.add({ targets: rec, y: H * 0.26 - 6, duration: 1800, yoyo: true, repeat: -1, ease: 'Sine.easeInOut' });
        }

        const scoreY = this.isNewRecord ? H * 0.36 : H * 0.32;

        this.add.text(cx, scoreY, 'Score', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.026) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5);

        this.add.text(cx, scoreY + Math.floor(H * 0.07), String(this.finalScore), {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.082) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#0088bb', blur: 30, fill: true },
        }).setOrigin(0.5);

        const bestY = scoreY + Math.floor(H * 0.185);
        this.add.text(cx, bestY, 'Best', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.022) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5);

        this.add.text(cx, bestY + Math.floor(H * 0.05), String(this.bestScore), {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.048) + 'px',
            fontStyle: 'bold',
            color: this.isNewRecord ? '#ffd740' : '#4a7a9a',
            shadow: this.isNewRecord
                ? { offsetX: 0, offsetY: 0, color: '#ff8800', blur: 16, fill: true }
                : undefined,
        }).setOrigin(0.5);

        const modeLabel = this.mode === 'human' ? 'Player' : this.agent.charAt(0).toUpperCase() + this.agent.slice(1) + ' Agent';
        this.add.text(cx, bestY + Math.floor(H * 0.105), modeLabel, {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.018) + 'px',
            color: '#1a3a5a',
        }).setOrigin(0.5);

        this.makeButton(cx, H * 0.74, 'Play Again', H, () => {
            this.scene.start('GameScene', { mode: this.mode, speed: 3.0, agent: this.agent });
        });

        this.makeButton(cx, H * 0.86, 'Main Menu', H, () => {
            this.scene.start('MenuScene');
        });

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => this.scene.restart(), 200);
    }

    makeButton(x, y, label, H, callback) {
        const W = Math.floor(H * 0.38);
        const Ht = Math.floor(H * 0.078);
        const R = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.028);

        const gfx = this.add.graphics();
        this.drawBtn(gfx, x, y, W, Ht, R, false);

        const txt = this.add.text(x, y, label, {
            fontFamily: 'Orbitron, Arial',
            fontSize: fs + 'px',
            fontStyle: 'bold',
            color: '#c8e8ff',
        }).setOrigin(0.5);
        fitTextsToWidth([txt], W);

        const zone = this.add.zone(x - W / 2, y - Ht / 2, W, Ht)
            .setOrigin(0, 0)
            .setInteractive({ useHandCursor: true });

        zone.on('pointerover', () => { this.drawBtn(gfx, x, y, W, Ht, R, true); txt.setStyle({ color: '#00d4ff' }); });
        zone.on('pointerout',  () => { this.drawBtn(gfx, x, y, W, Ht, R, false); txt.setStyle({ color: '#c8e8ff' }); });
        zone.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            this.time.delayedCall(140, callback);
        });
    }

    drawBtn(gfx, x, y, W, H, R, hover) {
        gfx.clear();
        gfx.fillStyle(hover ? 0x0a2060 : 0x060d28);
        gfx.fillRoundedRect(x - W / 2, y - H / 2, W, H, R);
        gfx.lineStyle(2, hover ? 0x00d4ff : 0x1a4888);
        gfx.strokeRoundedRect(x - W / 2, y - H / 2, W, H, R);
    }
}

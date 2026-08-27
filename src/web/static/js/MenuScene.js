class MenuScene extends Phaser.Scene {
    constructor() {
        super({ key: 'MenuScene' });
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

        this.spawnBackground(W, H);
        this.buildTitle(cx, H);
        this.buildButtons(cx, H);
        this.buildScoreBoard(W, H);

        this.add.text(cx, H - 30, 'University of Nottingham  ·  COMP4026', {
            fontFamily: 'Orbitron, Arial',
            fontSize: '16px',
            color: '#1a2a40',
        }).setOrigin(0.5, 1);

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => this.scene.restart(), 200);
    }

    spawnBackground(W, H) {
        const colors = [0x448aff, 0xff6d00, 0xffea00, 0xff1744, 0x00e676, 0xe040fb, 0x7c4dff, 0x00e5ff];
        this.bgBlocks = [];

        const count = Math.floor((W * H) / 30000);

        for (let i = 0; i < count; i++) {
            const size = Phaser.Math.Between(36, Math.floor(W * 0.08));
            const x = Phaser.Math.Between(-80, W + 80);
            const y = Phaser.Math.Between(-80, H + 80);
            const color = colors[i % colors.length];
            const alpha = Phaser.Math.FloatBetween(0.05, 0.14);

            const rect = this.add.rectangle(x, y, size, size, color, alpha);
            rect.rotation = Phaser.Math.FloatBetween(0, Math.PI * 2);

            this.bgBlocks.push({
                rect,
                vx: Phaser.Math.FloatBetween(-0.5, 0.5),
                vy: Phaser.Math.FloatBetween(-0.9, -0.2),
                rs: Phaser.Math.FloatBetween(-0.004, 0.004),
                W, H,
            });
        }
    }

    buildTitle(cx, H) {
        const titleSize = Math.floor(Math.min(H * 0.14, 120));

        this.add.text(cx, H * 0.18 + 4, 'BlastMind', {
            fontFamily: 'Orbitron, Arial',
            fontSize: titleSize + 'px',
            fontStyle: 'bold',
            color: '#003a60',
        }).setOrigin(0.5).setAlpha(0.5);

        const title = this.add.text(cx, H * 0.18, 'BlastMind', {
            fontFamily: 'Orbitron, Arial',
            fontSize: titleSize + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#00aaff', blur: 50, fill: true },
        }).setOrigin(0.5);

        this.tweens.add({
            targets: title,
            y: H * 0.18 - 12,
            duration: 2800,
            yoyo: true,
            repeat: -1,
            ease: 'Sine.easeInOut',
        });

        this.add.text(cx, H * 0.31, 'Block Blast  ·  AI Research Project', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.022) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5);
    }

    buildButtons(cx, H) {
        this.makeButton(cx, H * 0.47, 'Play Game', H, () => {
            this.scene.start('GameScene', { mode: 'human' });
        });

        this.makeButton(cx, H * 0.60, 'Watch AI', H, () => {
            this.scene.start('AgentSelectScene', { from: 'MenuScene' });
        });

        this.makeButton(cx, H * 0.73, 'Settings', H, () => {
            this.scene.start('SettingsScene', { from: 'MenuScene' });
        });

        this.makeButton(cx, H * 0.86, 'How to Play', H, () => {
            this.scene.start('HelpScene', { from: 'MenuScene' });
        });
    }

    buildScoreBoard(W, H) {
        const entries = [
            { label: 'Player',      key: 'human' },
            { label: 'Random',      key: 'random' },
            { label: 'Greedy',      key: 'greedy' },
            { label: 'Beam Search', key: 'beam' },
            { label: 'DQN',         key: 'dqn' },
            { label: 'MCTS',        key: 'mcts' },
        ];

        const rowH   = Math.floor(H * 0.036);
        const fs     = Math.floor(H * 0.019);
        const titleFS = Math.floor(H * 0.022);
        const padX   = Math.floor(W * 0.022);
        const padY   = Math.floor(H * 0.018);
        const innerW = Math.floor(Math.min(W * 0.22, 250));
        const innerH = titleFS + Math.floor(H * 0.02) + entries.length * rowH + padY;
        const panelW = innerW + padX * 2;
        const panelH = innerH + padY * 2;
        const panelX = W - Math.floor(W * 0.03) - panelW;
        const panelY = Math.floor(H * 0.08);

        const gfx = this.add.graphics();
        gfx.fillStyle(0x06091a, 0.82);
        gfx.fillRoundedRect(panelX, panelY, panelW, panelH, 10);
        gfx.lineStyle(1, 0x0d2244, 1);
        gfx.strokeRoundedRect(panelX, panelY, panelW, panelH, 10);

        const cx = panelX + panelW / 2;
        let y = panelY + padY;

        this.add.text(cx, y, 'Best Scores', {
            fontFamily: 'Orbitron, Arial',
            fontSize: titleFS + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
        }).setOrigin(0.5, 0);
        y += titleFS + Math.floor(H * 0.02);

        entries.forEach(({ label, key }) => {
            const score    = parseInt(localStorage.getItem('highScore_' + key) || '0', 10);
            const hasScore = score > 0;

            const labelTxt = this.add.text(panelX + padX, y + rowH / 2, label, {
                fontFamily: 'Orbitron, Arial',
                fontSize: fs + 'px',
                color: hasScore ? '#c8e8ff' : '#2a4a6a',
            }).setOrigin(0, 0.5);

            const scoreTxt = this.add.text(panelX + panelW - padX, y + rowH / 2, hasScore ? String(score) : '—', {
                fontFamily: 'Orbitron, Arial',
                fontSize: fs + 'px',
                fontStyle: 'bold',
                color: hasScore ? '#ffd740' : '#1a3050',
            }).setOrigin(1, 0.5);

            const maxScoreW = innerW - labelTxt.width - Math.floor(fs * 0.7);
            if (scoreTxt.width > maxScoreW) {
                scoreTxt.setScale(maxScoreW / scoreTxt.width);
            }

            y += rowH;
        });
    }

    makeButton(x, y, label, H, callback) {
        const W = Math.floor(H * 0.42);
        const Ht = Math.floor(H * 0.082);
        const R = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.030);

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

        zone.on('pointerover', () => {
            this.drawBtn(gfx, x, y, W, Ht, R, true);
            txt.setStyle({ color: '#00d4ff' });
        });
        zone.on('pointerout', () => {
            this.drawBtn(gfx, x, y, W, Ht, R, false);
            txt.setStyle({ color: '#c8e8ff' });
        });
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

    update() {
        this.bgBlocks.forEach(b => {
            b.rect.x += b.vx;
            b.rect.y += b.vy;
            b.rect.rotation += b.rs;
            if (b.rect.y < -200) {
                b.rect.y = b.H + 100;
                b.rect.x = Phaser.Math.Between(-80, b.W + 80);
            }
            if (b.rect.x < -200) b.rect.x = b.W + 100;
            if (b.rect.x > b.W + 200) b.rect.x = -100;
        });
    }
}

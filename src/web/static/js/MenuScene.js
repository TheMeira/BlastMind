class MenuScene extends Phaser.Scene {
    constructor() {
        super({ key: 'MenuScene' });
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');

        const W = this.scale.width;
        const H = this.scale.height;
        const cx = W / 2;

        this.spawnBackground(W, H);
        this.buildTitle(cx, H);
        this.buildButtons(cx, H);

        this.add.text(cx, H - 30, 'University of Nottingham  ·  COMP4026', {
            fontFamily: 'Orbitron, Arial',
            fontSize: '16px',
            color: '#1a2a40',
        }).setOrigin(0.5, 1);
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
        this.makeButton(cx, H * 0.50, 'Play Game', H, () => {
            this.scene.start('GameScene', { mode: 'human' });
        });

        this.makeButton(cx, H * 0.64, 'Watch AI (Random)', H, () => {
            this.scene.start('GameScene', { mode: 'ai', speed: 3.0, agent: 'random' });
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

class SettingsScene extends Phaser.Scene {
    constructor() {
        super({ key: 'SettingsScene' });
    }

    init(data) {
        this.fromScene = (data && data.from) || 'MenuScene';
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');

        const W = this.scale.width;
        const H = this.scale.height;
        const cx = W / 2;

        this.add.text(cx, H * 0.14, 'Settings', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(Math.min(H * 0.09, 80)) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#00aaff', blur: 40, fill: true },
        }).setOrigin(0.5);

        const sliderW = Math.floor(Math.min(W * 0.38, 420));

        this.makeSlider(cx, H * 0.38, sliderW, 'SFX Volume', 'sfxVolume', 0.6, H);
        this.makeSlider(cx, H * 0.58, sliderW, 'Music Volume', 'musicVolume', 0.6, H, true);

        this.makeButton(cx, H * 0.80, '← Back', H, () => {
            this.scene.start(this.fromScene);
        });

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    makeSlider(cx, y, w, label, storageKey, defaultVal, H, comingSoon = false) {
        const stored = parseFloat(localStorage.getItem(storageKey));
        let val = isNaN(stored) ? defaultVal : stored;

        const labelFS = Math.floor(H * 0.026);
        const valueFS = Math.floor(H * 0.024);

        this.add.text(cx - w / 2, y - Math.floor(H * 0.055), label, {
            fontFamily: 'Orbitron, Arial',
            fontSize: labelFS + 'px',
            fontStyle: 'bold',
            color: comingSoon ? '#1a3a5a' : '#c8e8ff',
        }).setOrigin(0, 0.5);

        if (comingSoon) {
            this.add.text(cx - w / 2 + Math.floor(labelFS * label.length * 0.62) + 12, y - Math.floor(H * 0.055), '(coming soon)', {
                fontFamily: 'Orbitron, Arial',
                fontSize: Math.floor(labelFS * 0.7) + 'px',
                color: '#1a3060',
            }).setOrigin(0, 0.5);
        }

        const trackColor = comingSoon ? 0x0d1830 : 0x1a3060;
        const fillColor = comingSoon ? 0x1a3060 : 0x00d4ff;
        const handleColor = comingSoon ? 0x1a3060 : 0x00d4ff;

        const track = this.add.graphics();
        track.fillStyle(trackColor);
        track.fillRoundedRect(cx - w / 2, y - 4, w, 8, 4);

        const fill = this.add.graphics();
        const redrawFill = (v) => {
            fill.clear();
            fill.fillStyle(fillColor);
            fill.fillRoundedRect(cx - w / 2, y - 4, Math.max(8, w * v), 8, 4);
        };
        redrawFill(val);

        const handle = this.add.circle(cx - w / 2 + w * val, y, 13, handleColor);

        const valTxt = this.add.text(cx + w / 2 + 16, y, Math.round(val * 100) + '%', {
            fontFamily: 'Orbitron, Arial',
            fontSize: valueFS + 'px',
            color: comingSoon ? '#1a3a5a' : '#00d4ff',
        }).setOrigin(0, 0.5);

        if (!comingSoon) {
            handle.setInteractive({ useHandCursor: true, draggable: true });
            handle.on('drag', (ptr, dragX) => {
                const clamped = Phaser.Math.Clamp(dragX, cx - w / 2, cx + w / 2);
                handle.x = clamped;
                const newVal = (clamped - (cx - w / 2)) / w;
                redrawFill(newVal);
                valTxt.setText(Math.round(newVal * 100) + '%');
                localStorage.setItem(storageKey, String(newVal));
            });
        }
    }

    makeButton(x, y, label, H, callback) {
        const W = Math.floor(H * 0.38);
        const Ht = Math.floor(H * 0.080);
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

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => this.scene.restart({ from: this.fromScene }), 200);
    }
}
